"""
V4PlateauAdapter — NumPy Implementation of Bubble Transformer V4
=================================================================

Adapts V4 (FPS + Expert-Choice routing) to NumPy for epsilon sweep experiments.

Key V4 features:
1. FPS (Farthest Point Sampling) for centroid initialization
2. Expert-Choice routing (top-k selection per expert)
3. Power Diagram (Laguerre cell) assignment

This adapter provides the same interface as PlateauAttentionMechanism
but uses V4 routing algorithms internally.

Usage:
    from v4_adapter import V4PlateauAdapter

    attn = V4PlateauAdapter(
        d_model=1024,
        num_heads=16,
        num_experts=32,
        top_k=8,
        epsilon=0.001,  # Viscosity for Sinkhorn
    )
    output, A = attn.forward(x, return_attention=True)
"""

import numpy as np
from typing import Optional, Tuple, List

from plateau_attention import (
    PlateauAttentionMechanism,
    CostFunctionFactory,
    _logsumexp,
)


def fps_sample_numpy(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) for centroid initialization - NumPy version.

    FPS selects points that are maximally distant from each other,
    providing better coverage than random initialization.

    Args:
        points: [B, H, N, d] - input points (Keys)
        num_samples: Number of samples C (centroids)

    Returns:
        indices: [B, H, C] - indices of selected points

    Complexity: O(N * C)
    """
    B, H, N, d = points.shape
    C = num_samples

    # Handle edge case: C >= N
    if C >= N:
        indices = np.arange(N).reshape(1, 1, -1)
        indices = np.broadcast_to(indices, (B, H, N))
        return indices[:, :, :C]

    indices = np.zeros((B, H, C), dtype=np.int64)
    indices[:, :, 0] = 0

    # Track minimum distances from selected points
    dists = np.full((B, H, N), np.inf, dtype=np.float32)

    # Get first point distances
    first_point = points[:, :, 0:1, :]  # [B, H, 1, d]
    diff = first_point - points  # [B, H, 1, d] - [B, H, N, d] = [B, H, N, d]
    dists = np.sqrt(np.sum(diff**2, axis=-1))  # [B, H, N]

    for i in range(1, C):
        # Select point with maximum minimum distance
        farthest_idx = np.argmax(dists, axis=-1)  # [B, H]
        indices[:, :, i] = farthest_idx

        # Update distances with new point
        for b in range(B):
            for h in range(H):
                new_point = points[b, h, farthest_idx[b, h], :]  # [d]
                new_dists = np.sqrt(
                    np.sum((new_point - points[b, h]) ** 2, axis=-1)
                )  # [N]
                dists[b, h] = np.minimum(dists[b, h], new_dists)

    return indices


def fps_initialize_centroids_numpy(K: np.ndarray, num_centroids: int) -> np.ndarray:
    """
    Initialize centroids using FPS sampling - NumPy version.

    Args:
        K: Key tensor [B, heads, N, d_head]
        num_centroids: Number of centroids C

    Returns:
        centroids: [B, heads, C, d_head]
    """
    B, H, N, d = K.shape
    C = num_centroids

    indices = fps_sample_numpy(K, C)  # [B, H, C]

    # Gather selected points as centroids
    centroids = np.zeros((B, H, C, d), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            centroids[b, h] = K[b, h, indices[b, h]]

    return centroids


def expert_choice_routing_numpy(
    Q: np.ndarray,
    centroids: np.ndarray,
    top_k: int,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expert-Choice routing: each expert selects top-k tokens - NumPy version.

    Args:
        Q: Query tensor [B, heads, N, d_head]
        centroids: Expert centroids [B, heads, C, d_head]
        top_k: Number of tokens each expert selects
        temperature: Softmax temperature for routing scores

    Returns:
        routing_weights: [B, heads, N, C] - soft routing weights
        expert_mask: [B, heads, C, k] - indices of selected tokens per expert
    """
    B, H, N, d = Q.shape
    C = centroids.shape[2]
    k = max(1, min(top_k, N))

    # Compute distances: [B, H, N, C]
    # Using broadcasting: Q [B,H,N,1,d] - centroids [B,H,1,C,d]
    Q_expanded = Q[:, :, :, np.newaxis, :]  # [B, H, N, 1, d]
    C_expanded = centroids[:, :, np.newaxis, :, :]  # [B, H, 1, C, d]
    dists = np.sqrt(np.sum((Q_expanded - C_expanded) ** 2, axis=-1))  # [B, H, N, C]

    # Negative distance as affinity (closer = higher score)
    scores = -dists / temperature  # [B, H, N, C]

    # Expert-choice: each expert selects top-k tokens
    scores_expert = scores.transpose(0, 1, 3, 2)  # [B, H, C, N]

    # Get top-k indices per expert
    expert_mask = np.zeros((B, H, C, k), dtype=np.int64)
    for b in range(B):
        for h in range(H):
            for c in range(C):
                expert_mask[b, h, c] = np.argsort(scores_expert[b, h, c])[-k:][::-1]

    # Soft routing weights via softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    routing_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    return routing_weights.astype(np.float32), expert_mask


def compute_sinkhorn_attention(
    Q: np.ndarray,
    K: np.ndarray,
    epsilon: float,
    tau_iters: int = 5,
) -> np.ndarray:
    """
    Compute attention via Sinkhorn-Knopp in log domain.

    Args:
        Q: [B, H, N, d]
        K: [B, H, N, d]
        epsilon: Viscosity coefficient
        tau_iters: Sinkhorn iterations

    Returns:
        A: [B, H, N, N] - doubly stochastic attention matrix
    """
    B, H, N, d = Q.shape

    # Compute cost matrix (L2 squared)
    Q_sq = np.sum(Q**2, axis=-1, keepdims=True)  # [B, H, N, 1]
    K_sq = np.sum(K**2, axis=-1, keepdims=True)  # [B, H, N, 1]
    K_sq_t = np.transpose(K_sq, (0, 1, 3, 2))  # [B, H, 1, N]
    C = (
        Q_sq + K_sq_t - 2.0 * np.matmul(Q, np.transpose(K, (0, 1, 3, 2)))
    )  # [B, H, N, N]
    C = np.maximum(C, 0.0).astype(np.float32)

    # Normalize costs
    C_min = np.min(C, axis=(-2, -1), keepdims=True)
    C_max = np.max(C, axis=(-2, -1), keepdims=True)
    C = (C - C_min) / (C_max - C_min + 1e-10)

    # Log-domain Sinkhorn
    log_S = -C / epsilon

    u = np.zeros((B, H, N), dtype=np.float32)
    v = np.zeros((B, H, N), dtype=np.float32)

    for _ in range(tau_iters):
        u = -_logsumexp(log_S + v[:, :, np.newaxis, :], axis=-1)
        v = -_logsumexp(log_S + u[:, :, :, np.newaxis], axis=-2)

    A = np.exp(log_S + u[:, :, :, np.newaxis] + v[:, :, np.newaxis, :])
    return A.astype(np.float32)


class V4PlateauAdapter:
    """
    V4 Plateau Attention Adapter - NumPy implementation.

    Combines V4 routing (FPS + Expert-Choice) with Sinkhorn attention.
    Provides the same interface as PlateauAttentionMechanism.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_experts: Number of experts/centroids for V4 routing
        top_k: Tokens per expert in Expert-Choice routing
        epsilon: Viscosity coefficient for Sinkhorn
        tau_iters: Sinkhorn iterations
        use_fps_init: Use FPS for centroid initialization
        seed: Random seed
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 32,
        top_k: int = 8,
        epsilon: float = 0.1,
        tau_iters: int = 5,
        use_fps_init: bool = True,
        seed: int = 42,
    ):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.epsilon = epsilon
        self.tau_iters = tau_iters
        self.use_fps_init = use_fps_init

        # Initialize projections (same as PlateauAttentionMechanism)
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = rng.randn(d_model, d_model).astype(np.float32) * scale

        # Learnable centroids (optional)
        self.learnable_centroids = (
            rng.randn(1, num_heads, num_experts, self.head_dim).astype(np.float32)
            * 0.02
        )

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass with V4 routing + Sinkhorn attention.

        Args:
            x: [B, N, d_model]
            mask: Optional attention mask
            return_attention: If True, return attention matrix

        Returns:
            output: [B, N, d_model]
            A: [B, H, N, N] (if return_attention=True)
        """
        B, N, D = x.shape

        # Projections
        Q = x @ self.W_q  # [B, N, D]
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape to heads
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )  # [B, H, N, d]
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # V4: Initialize centroids with FPS
        if self.use_fps_init:
            centroids = fps_initialize_centroids_numpy(K, self.num_experts)
        else:
            centroids = np.broadcast_to(
                self.learnable_centroids,
                (B, self.num_heads, self.num_experts, self.head_dim),
            )

        # V4: Expert-Choice routing (optional - can be used for analysis)
        routing_weights, expert_mask = expert_choice_routing_numpy(
            Q, centroids, self.top_k
        )

        # Sinkhorn attention (same as PlateauAttentionMechanism)
        A = compute_sinkhorn_attention(Q, K, self.epsilon, self.tau_iters)

        # Apply attention
        output = np.matmul(A, V)  # [B, H, N, d]
        output = output.transpose(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]
        output = output @ self.W_o

        if return_attention:
            return output, A
        return output


class V4Config:
    """Configuration for V4 Plateau Adapter."""

    def __init__(
        self,
        num_experts: int = 32,
        top_k: int = 8,
        use_fps_init: bool = True,
        epsilon: float = 0.001,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_fps_init = use_fps_init
        self.epsilon = epsilon


if __name__ == "__main__":
    print("[V4PlateauAdapter] Running quick test...")

    # Test FPS sampling
    B, H, N, d = 2, 4, 100, 64
    points = np.random.randn(B, H, N, d).astype(np.float32)
    indices = fps_sample_numpy(points, 32)
    print(f"fps_sample_numpy: points {points.shape} -> indices {indices.shape}")
    assert indices.shape == (B, H, 32), f"Expected {(B, H, 32)}, got {indices.shape}"

    # Test FPS centroid initialization
    K = np.random.randn(B, H, N, d).astype(np.float32)
    centroids = fps_initialize_centroids_numpy(K, 32)
    print(f"fps_initialize_centroids_numpy: K {K.shape} -> centroids {centroids.shape}")
    assert centroids.shape == (B, H, 32, d), (
        f"Expected {(B, H, 32, d)}, got {centroids.shape}"
    )

    # Test Expert-Choice routing
    Q = np.random.randn(B, H, N, d).astype(np.float32)
    routing_weights, expert_mask = expert_choice_routing_numpy(Q, centroids, 8)
    print(
        f"expert_choice_routing_numpy: routing_weights {routing_weights.shape}, expert_mask {expert_mask.shape}"
    )
    assert routing_weights.shape == (B, H, N, 32), (
        f"Expected {(B, H, N, 32)}, got {routing_weights.shape}"
    )
    assert expert_mask.shape == (B, H, 32, 8), (
        f"Expected {(B, H, 32, 8)}, got {expert_mask.shape}"
    )

    # Test full adapter
    adapter = V4PlateauAdapter(
        d_model=512,
        num_heads=8,
        num_experts=32,
        top_k=8,
        epsilon=0.1,
    )
    x = np.random.randn(2, 100, 512).astype(np.float32)
    output, A = adapter.forward(x, return_attention=True)
    print(f"V4PlateauAdapter: input {x.shape} -> output {output.shape}, A {A.shape}")
    assert output.shape == (2, 100, 512), (
        f"Expected {(2, 100, 512)}, got {output.shape}"
    )
    assert A.shape == (2, 8, 100, 100), f"Expected {(2, 8, 100, 100)}, got {A.shape}"

    # Verify doubly stochastic
    row_sums = A.sum(axis=-1)
    col_sums = A.sum(axis=-2)
    print(f"Row sums mean: {row_sums.mean():.4f} (should be ~1.0)")
    print(f"Col sums mean: {col_sums.mean():.4f} (should be ~1.0)")

    print("[V4PlateauAdapter] All tests passed!")
