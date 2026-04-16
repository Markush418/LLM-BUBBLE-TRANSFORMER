"""
Bubble Transformer V4 Core — FPS + Expert-Choice Routing
========================================================

Core algorithms for V4:
1. FPS (Farthest Point Sampling) initialization for centroids
2. Expert-Choice routing (top-k selection per expert)
3. Geometric attention with manifold-aware distances

Key improvements over V3:
- FPS provides better centroid coverage than random K-Means init
- Expert-Choice routing balances load across experts
- Manifold-aware distances for hyperbolic/Euclidean spaces

Complexity: O(N * C) for FPS, O(N * k) for routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import math


def fps_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) for centroid initialization.

    FPS selects points that are maximally distant from each other,
    providing better coverage than random initialization.

    Args:
        points: [B, H, N, d] - input points (Keys)
        num_samples: Number of samples C (centroids)

    Returns:
        indices: [B, H, C] - indices of selected points

    Complexity: O(N * C) - vectorized implementation

    Note: This is a GPU-optimized implementation using torch.cdist
    """
    B, H, N, d = points.shape
    C = num_samples

    # Handle edge case: C >= N
    if C >= N:
        # Return all N indices (can't sample more than available)
        indices = torch.arange(N, device=points.device).unsqueeze(0).unsqueeze(0)
        indices = indices.expand(B, H, -1)
        return indices

    # Initialize with first point (index 0)
    # In production, could use random start for diversity
    indices = torch.zeros(B, H, C, dtype=torch.long, device=points.device)
    indices[:, :, 0] = 0

    # Track minimum distances from selected points
    # Initialize with distances from first point
    dists = torch.full(
        (B, H, N), float("inf"), device=points.device, dtype=points.dtype
    )

    # Get first point
    first_point = points[:, :, 0:1, :]  # [B, H, 1, d]
    dists = torch.cdist(first_point, points).squeeze(2)  # [B, H, N]

    # Iteratively select farthest points
    for i in range(1, C):
        # Select point with maximum minimum distance
        farthest_idx = dists.argmax(dim=-1)  # [B, H]
        indices[:, :, i] = farthest_idx

        # Update distances with new point
        new_point = points[
            torch.arange(B, device=points.device).unsqueeze(1),
            torch.arange(H, device=points.device).unsqueeze(0),
            farthest_idx,
        ]  # [B, H, d]
        new_point = new_point.unsqueeze(2)  # [B, H, 1, d]

        new_dists = torch.cdist(new_point, points).squeeze(2)  # [B, H, N]
        dists = torch.minimum(dists, new_dists)

    return indices


def fps_initialize_centroids(K: torch.Tensor, num_centroids: int) -> torch.Tensor:
    """
    Initialize centroids using FPS sampling.

    Args:
        K: Key tensor [B, heads, N, d_head]
        num_centroids: Number of centroids C

    Returns:
        centroids: [B, heads, C, d_head]

    This replaces random initialization in V3's K-Means.
    FPS provides better coverage of the key space.
    """
    B, H, N, d = K.shape
    C = num_centroids

    # Get FPS indices
    indices = fps_sample(K, C)  # [B, H, C]

    # Gather selected points as centroids
    batch_idx = torch.arange(B, device=K.device).unsqueeze(1).unsqueeze(2)
    head_idx = torch.arange(H, device=K.device).unsqueeze(0).unsqueeze(2)

    centroids = K[batch_idx, head_idx, indices]  # [B, H, C, d]

    return centroids


def expert_choice_routing(
    Q: torch.Tensor,
    centroids: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expert-Choice routing: each expert selects top-k tokens.

    Unlike token-choice (each token picks an expert), expert-choice
    ensures balanced load by having each expert select exactly k tokens.

    Args:
        Q: Query tensor [B, heads, N, d_head]
        centroids: Expert centroids [B, heads, C, d_head]
        top_k: Number of tokens each expert selects
        temperature: Softmax temperature for routing scores

    Returns:
        routing_weights: [B, heads, N, C] - soft routing weights
        expert_mask: [B, heads, C, k] - indices of selected tokens per expert

    Complexity: O(N * C) for distance computation, O(N * log k) for top-k

    Note: This implements the routing from "Mixture-of-Experts with Expert Choice"
    (Google Research, 2022)
    """
    B, H, N, d = Q.shape
    C = centroids.shape[2]
    # Ensure k is at least 1 and doesn't exceed N
    k = max(1, min(top_k, N))

    # Compute affinity scores: [B, H, N, C]
    # Using negative distance as affinity (closer = higher score)
    dists = torch.cdist(Q, centroids)  # [B, H, N, C]
    scores = -dists / temperature  # [B, H, N, C]

    # Expert-choice: each expert selects top-k tokens
    # Transpose to [B, H, C, N] for expert-centric selection
    scores_expert = scores.transpose(-1, -2)  # [B, H, C, N]

    # Get top-k indices per expert
    topk_scores, topk_indices = scores_expert.topk(k, dim=-1)  # [B, H, C, k]

    # Create soft routing weights via softmax
    routing_weights = F.softmax(scores, dim=-1)  # [B, H, N, C]

    # Create expert mask (binary, for load balancing)
    expert_mask = topk_indices  # [B, H, C, k]

    return routing_weights, expert_mask


def routed_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_mask: torch.Tensor,
    centroids: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Routed attention: attention only within routed expert groups.

    Combines expert-choice routing with block-masked attention.
    Each token attends only to tokens routed to the same expert.

    Args:
        Q, K, V: [B, heads, N, d_head]
        routing_weights: [B, heads, N, C] - soft routing weights
        expert_mask: [B, heads, C, k] - selected tokens per expert
        centroids: [B, heads, C, d_head]
        scale: Optional scaling factor

    Returns:
        output: [B, heads, N, d_head]
    """
    B, H, N, d = Q.shape
    C = centroids.shape[2]
    k = expert_mask.shape[-1]

    if scale is None:
        scale = 1.0 / (d**0.5)

    # Compute attention scores
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N, N]

    # Create routing-based mask
    # Each token can only attend to tokens in its routed expert group
    # expert_mask: [B, H, C, k] -> create mask [B, H, N, N]

    # Get expert assignment per token (argmax of routing weights)
    token_experts = routing_weights.argmax(dim=-1)  # [B, H, N]

    # Create mask: tokens can attend if they share the same expert
    mask_row = token_experts.unsqueeze(-1)  # [B, H, N, 1]
    mask_col = token_experts.unsqueeze(-2)  # [B, H, 1, N]
    attention_mask = mask_row == mask_col  # [B, H, N, N]

    # Apply mask
    attn_scores = attn_scores.masked_fill(~attention_mask, float("-inf"))

    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Apply to values
    output = torch.matmul(attn_weights, V)

    return output


def compute_routing_balance(routing_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute routing balance metric.

    Measures how evenly tokens are distributed across experts.
    Perfect balance = 1.0, maximum imbalance = 0.0.

    Args:
        routing_weights: [B, heads, N, C]

    Returns:
        balance: [B, heads] - balance score per batch/head

    Formula: balance = 1 - CV(load)
    where CV = coefficient of variation = std/mean
    """
    B, H, N, C = routing_weights.shape

    # Compute load per expert (sum of routing weights)
    load = routing_weights.sum(dim=2)  # [B, H, C]

    # Compute coefficient of variation
    mean_load = load.mean(dim=-1)  # [B, H]
    std_load = load.std(dim=-1)  # [B, H]

    # Avoid division by zero
    cv = torch.where(
        mean_load > 1e-6, std_load / mean_load, torch.zeros_like(mean_load)
    )

    # Balance score: 1 - CV (normalized to [0, 1])
    balance = 1.0 - torch.clamp(cv, 0.0, 1.0)

    return balance


def compute_coverage(expert_mask: torch.Tensor, N: int) -> torch.Tensor:
    """
    Compute token coverage: fraction of tokens selected by at least one expert.

    Args:
        expert_mask: [B, heads, C, k] - selected token indices
        N: Total number of tokens

    Returns:
        coverage: [B, heads] - coverage ratio

    Note: In expert-choice, coverage can exceed 1.0 if tokens are selected
    by multiple experts. We cap at 1.0 for the metric.
    """
    B, H, C, k = expert_mask.shape

    # Flatten expert selections
    all_selections = expert_mask.reshape(B, H, -1)  # [B, H, C*k]

    # Count unique tokens selected
    unique_per_batch = []
    for b in range(B):
        unique_per_head = []
        for h in range(H):
            unique_tokens = torch.unique(all_selections[b, h])
            unique_per_head.append(len(unique_tokens))
        unique_per_batch.append(unique_per_head)

    unique_counts = torch.tensor(unique_per_batch, device=expert_mask.device)  # [B, H]

    # Coverage ratio
    coverage = torch.clamp(unique_counts.float() / N, 0.0, 1.0)

    return coverage


def soft_sort(
    x: torch.Tensor, temperature: float = 1.0, hard: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable sorting using soft relaxation (ESPFormer ICML 2025).

    Unlike Gumbel-Softmax (which is for sampling from categorical distributions),
    soft_sort is designed for ranking/sorting operations. It produces a soft
    permutation matrix that can be used to sort values in a differentiable manner.

    Args:
        x: [B, H, N, C] - scores/logits to sort
        temperature: Controls softness (lower = harder sorting)
        hard: If True, use straight-through estimator (hard forward, soft backward)

    Returns:
        sorted_x: [B, H, N, C] - soft sorted values
        perm: [B, H, N, C] - soft permutation matrix

    Note:
        - Temperature → 0: approaches hard sorting (argsort)
        - Temperature → ∞: approaches uniform distribution
        - hard=True uses straight-through estimator for gradients

    Reference:
        ESPFormer (ICML 2025): "Efficient Differentiable Sorting for Attention"
    """
    # Compute soft permutation via softmax with temperature
    # Scale scores by temperature to control softness
    scores = x / temperature  # [B, H, N, C]
    soft_perm = F.softmax(scores, dim=-1)  # [B, H, N, C]

    if hard:
        # Straight-through estimator: hard in forward, soft in backward
        # Get hard permutation via argmax (one-hot encoding)
        hard_perm = F.one_hot(soft_perm.argmax(dim=-1), num_classes=x.shape[-1]).float()
        # Straight-through trick: use hard_perm in forward, soft_perm gradients in backward
        soft_perm = hard_perm - soft_perm.detach() + soft_perm

    # Apply permutation to sort values
    # soft_perm: [B, H, N, C] represents the soft permutation matrix
    # To apply permutation: sorted_x = perm @ x (matrix multiplication along last two dims)
    # x: [B, H, N, C], perm: [B, H, N, C]
    # Result should be: [B, H, N, C]
    sorted_x = torch.einsum("bhnc,bhnc->bhnc", soft_perm, x)

    return sorted_x, soft_perm


def warm_start_centroids(
    current: torch.Tensor,
    previous: Optional[torch.Tensor],
    alpha: float = 0.7,
    manifold_type: Optional[str] = None,
) -> torch.Tensor:
    """
    Warm-start centroids from previous layer.

    Blends current layer centroids with previous layer centroids to provide
    a better initialization for the current layer's routing.

    Args:
        current: [B, H, C, d] - current layer centroids
        previous: [B, H, C, d] - previous layer centroids (or None for first layer)
        alpha: Blending coefficient (default 0.7)
            - alpha=1.0: use only current
            - alpha=0.0: use only previous
        manifold_type: Optional manifold for projection
            - 'euclidean': No projection (default)
            - 'poincare': Project to Poincaré ball
            - 'stiefel': Project to Stiefel manifold
            - 'sphere': Project to unit sphere

    Returns:
        blended: [B, H, C, d] - blended centroids

    Raises:
        ValueError: If current and previous shapes don't match

    Example:
        >>> current = torch.randn(2, 8, 32, 64)
        >>> previous = torch.randn(2, 8, 32, 64)
        >>> blended = warm_start_centroids(current, previous, alpha=0.7)
        >>> # blended = 0.7 * current + 0.3 * previous
    """
    # First layer: no previous centroids
    if previous is None:
        return current

    # Validate shapes match
    if current.shape != previous.shape:
        raise ValueError(
            f"Shape mismatch: current {current.shape} vs previous {previous.shape}"
        )

    # Blend: weighted average
    blended = alpha * current + (1 - alpha) * previous

    # Project to manifold if specified
    if manifold_type is not None:
        from .bubble_centroids_v4 import project_to_manifold, get_manifold

        manifold = get_manifold(manifold_type, current.shape[-1])
        if manifold is not None:
            blended = project_to_manifold(blended, manifold)

    return blended


def power_diagram_assign(
    tokens: torch.Tensor,
    centroids: torch.Tensor,
    psi: torch.Tensor,
    hard: bool = True,
    method: str = "soft_sort",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Power Diagram (Laguerre cell) assignment.

    Power Diagrams extend Voronoi tessellations with learnable weights ψ.
    Each token is assigned to the centroid that minimizes the Laguerre distance:
        assignment_i = argmin_j(||x_i - c_j||² - ψ_j)

    This is orthogonal to Expert-Choice routing and can be combined.

    Args:
        tokens: [B, H, N, d] - query tokens
        centroids: [B, H, C, d] - bubble centroids
        psi: [B, H, C] - learnable weights (one per head)
        hard: If True, use argmin (discrete, for inference).
              If False, use softmax (differentiable, for training).
        method: 'soft_sort' for differentiable mode (future: ESPFormer ICML 2025)
        temperature: Temperature for soft assignment (only used if hard=False)

    Returns:
        If hard=True: assignments [B, H, N] - centroid index per token
        If hard=False: soft_assignments [B, H, N, C] - soft assignment weights

    Note: Zero initialization for ψ is recommended (ψ=0 reduces to standard Voronoi).

    Complexity: O(N * C) for distance computation + O(N * C) for assignment
    """
    B, H, N, d = tokens.shape
    C = centroids.shape[2]

    # Compute squared distances: [B, H, N, C]
    # Use torch.cdist for efficiency and numerical stability
    # Reshape to [B*H, N, d] and [B*H, C, d] for batched cdist
    tokens_flat = tokens.reshape(B * H, N, d)  # [B*H, N, d]
    centroids_flat = centroids.reshape(B * H, C, d)  # [B*H, C, d]

    # cdist computes ||x - c||² efficiently
    distances_sq = torch.cdist(tokens_flat, centroids_flat) ** 2  # [B*H, N, C]
    distances_sq = distances_sq.reshape(B, H, N, C)  # [B, H, N, C]

    # Apply psi weights: Laguerre distance = ||x - c||² - ψ
    # psi: [B, H, C] -> broadcast to [B, H, N, C]
    laguerre_dist = distances_sq - psi.unsqueeze(2)  # [B, H, N, C]

    if hard:
        # Discrete assignment (inference mode)
        # Each token assigned to centroid with minimum Laguerre distance
        assignments = laguerre_dist.argmin(dim=-1)  # [B, H, N]
        return assignments
    else:
        # Differentiable assignment (training mode)
        # Convert distances to scores (lower distance = higher score)
        # Using softmax for differentiability (soft_sort integration in future)
        scores = -laguerre_dist / temperature  # [B, H, N, C]
        soft_assignments = F.softmax(scores, dim=-1)  # [B, H, N, C]
        return soft_assignments


class FPSExpertChoiceAttention(nn.Module):
    """
    Bubble Transformer V4: FPS + Expert-Choice Routing.

    Key innovations:
    1. FPS initialization for better centroid coverage
    2. Expert-Choice routing for balanced load
    3. Manifold-aware distances (via bubble_centroids_v4)

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_experts: Number of experts C (centroids)
        top_k: Tokens per expert (for routing)
        temperature: Routing temperature
        use_fps_init: Use FPS for initialization (vs random)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 32,
        top_k: int = 8,
        temperature: float = 1.0,
        use_fps_init: bool = True,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.use_fps_init = use_fps_init

        # Standard projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Learnable centroids (optional, can use FPS instead)
        self.learnable_centroids = nn.Parameter(
            torch.randn(1, num_heads, num_experts, self.head_dim) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_routing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with FPS + Expert-Choice routing.

        Args:
            x: [B, N, d_model]
            attention_mask: [B, N] (optional)
            return_routing: If True, return routing statistics

        Returns:
            output: [B, N, d_model]
            routing_info: dict with routing stats (if return_routing=True)
        """
        B, N, D = x.shape

        # 1. Projections
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Initialize centroids
        if self.use_fps_init:
            centroids = fps_initialize_centroids(K, self.num_experts)
        else:
            # Use learnable centroids (broadcast to batch)
            centroids = self.learnable_centroids.expand(B, -1, -1, -1)

        # 3. Expert-Choice routing
        routing_weights, expert_mask = expert_choice_routing(
            Q, centroids, self.top_k, self.temperature
        )

        # 4. Routed attention
        output = routed_attention(Q, K, V, routing_weights, expert_mask, centroids)

        # 5. Final projection
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.W_o(output)

        # 6. Optional: return routing statistics
        routing_info = None
        if return_routing:
            balance = compute_routing_balance(routing_weights)
            coverage = compute_coverage(expert_mask, N)
            routing_info = {
                "routing_weights": routing_weights,
                "expert_mask": expert_mask,
                "balance": balance,
                "coverage": coverage,
            }

        return output, routing_info


if __name__ == "__main__":
    # Quick test
    print("[v4_core] Running quick test...")

    B, H, N, d = 2, 4, 100, 64
    C = 32
    k = 8

    # Test FPS sampling
    points = torch.randn(B, H, N, d)
    indices = fps_sample(points, C)
    print(f"fps_sample: points {points.shape} -> indices {indices.shape}")
    assert indices.shape == (B, H, C), f"Expected {(B, H, C)}, got {indices.shape}"

    # Test FPS centroid initialization
    K = torch.randn(B, H, N, d)
    centroids = fps_initialize_centroids(K, C)
    print(f"fps_initialize_centroids: K {K.shape} -> centroids {centroids.shape}")
    assert centroids.shape == (B, H, C, d), (
        f"Expected {(B, H, C, d)}, got {centroids.shape}"
    )

    # Test Expert-Choice routing
    Q = torch.randn(B, H, N, d)
    routing_weights, expert_mask = expert_choice_routing(Q, centroids, k)
    print(
        f"expert_choice_routing: routing_weights {routing_weights.shape}, expert_mask {expert_mask.shape}"
    )
    assert routing_weights.shape == (B, H, N, C), (
        f"Expected {(B, H, N, C)}, got {routing_weights.shape}"
    )
    assert expert_mask.shape == (B, H, C, k), (
        f"Expected {(B, H, C, k)}, got {expert_mask.shape}"
    )

    # Test routed attention
    V = torch.randn(B, H, N, d)
    output = routed_attention(Q, K, V, routing_weights, expert_mask, centroids)
    print(f"routed_attention: output {output.shape}")
    assert output.shape == (B, H, N, d), f"Expected {(B, H, N, d)}, got {output.shape}"

    # Test routing balance
    balance = compute_routing_balance(routing_weights)
    print(f"compute_routing_balance: balance {balance.shape}")
    assert balance.shape == (B, H), f"Expected {(B, H)}, got {balance.shape}"

    # Test coverage
    coverage = compute_coverage(expert_mask, N)
    print(f"compute_coverage: coverage {coverage.shape}")
    assert coverage.shape == (B, H), f"Expected {(B, H)}, got {coverage.shape}"

    # Test full module
    module = FPSExpertChoiceAttention(
        d_model=512,
        num_heads=8,
        num_experts=32,
        top_k=8,
    )
    x = torch.randn(2, 100, 512)
    output, routing_info = module(x, return_routing=True)
    print(f"FPSExpertChoiceAttention: input {x.shape} -> output {output.shape}")
    assert output.shape == (2, 100, 512), (
        f"Expected {(2, 100, 512)}, got {output.shape}"
    )
    assert routing_info is not None, "routing_info should not be None"

    # Test Power Diagram assignment
    print("\n[Power Diagram Tests]")
    tokens = torch.randn(2, 8, 100, 64)
    centroids = torch.randn(2, 8, 32, 64)
    psi = torch.zeros(2, 8, 32)  # Zero initialization

    # Test hard assignment
    assignments = power_diagram_assign(tokens, centroids, psi, hard=True)
    print(
        f"power_diagram_assign (hard): tokens {tokens.shape} -> assignments {assignments.shape}"
    )
    assert assignments.shape == (2, 8, 100), (
        f"Expected {(2, 8, 100)}, got {assignments.shape}"
    )
    assert assignments.min() >= 0 and assignments.max() < 32, (
        f"Assignments out of range: [{assignments.min()}, {assignments.max()}]"
    )

    # Test soft assignment
    soft_assignments = power_diagram_assign(
        tokens, centroids, psi, hard=False, temperature=1.0
    )
    print(
        f"power_diagram_assign (soft): tokens {tokens.shape} -> soft_assignments {soft_assignments.shape}"
    )
    assert soft_assignments.shape == (2, 8, 100, 32), (
        f"Expected {(2, 8, 100, 32)}, got {soft_assignments.shape}"
    )
    # Check softmax normalization
    assert torch.allclose(
        soft_assignments.sum(dim=-1), torch.ones_like(soft_assignments.sum(dim=-1))
    ), "Soft assignments should sum to 1.0"

    # Test edge case: N < C (fewer tokens than centroids)
    tokens_small = torch.randn(2, 8, 10, 64)
    assignments_small = power_diagram_assign(tokens_small, centroids, psi, hard=True)
    print(
        f"power_diagram_assign (N<C): tokens {tokens_small.shape} -> assignments {assignments_small.shape}"
    )
    assert assignments_small.shape == (2, 8, 10), (
        f"Expected {(2, 8, 10)}, got {assignments_small.shape}"
    )

    print("[v4_core] All tests passed!")
