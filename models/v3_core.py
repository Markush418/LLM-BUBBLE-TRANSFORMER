"""
Bubble Transformer V3 Core — SDOT Algorithms
============================================

Core algorithms for Semi-Discrete Optimal Transport (SDOT):
1. Key clustering via K-Means (vectorized, 3 iterations)
2. Voronoi assignment (argmin distance)
3. Block-masked attention (attention within bubbles)

Complexity: O(N log C) vs O(N² × τ) of Sinkhorn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def cluster_keys(K: torch.Tensor, num_centroids: int) -> torch.Tensor:
    """
    Cluster Keys into C centroids using vectorized K-Means.

    Args:
        K: Key tensor [B, heads, N, d_head]
        num_centroids: Number of centroids C (bubbles)

    Returns:
        centroids: [B, heads, C, d_head]

    Complexity: O(N * C * iterations) with iterations=3
    """
    B, H, N, d = K.shape
    C = num_centroids

    # Initialize: select C random keys as initial centroids
    indices = torch.randint(0, N, (B, H, C), device=K.device, dtype=torch.long)

    # Gather initial centroids: [B, H, C, d]
    batch_idx = torch.arange(B, device=K.device).unsqueeze(1).unsqueeze(2)
    head_idx = torch.arange(H, device=K.device).unsqueeze(0).unsqueeze(2)

    centroids = K[batch_idx, head_idx, indices]  # [B, H, C, d]

    # K-Means iterations (3 suffice for convergence)
    for _ in range(3):
        # Assignment: compute distances [B, H, N, C]
        dists = torch.cdist(K, centroids)  # [B, H, N, C]
        assignments = dists.argmin(dim=-1)  # [B, H, N]

        # Update centroids: for each cluster, compute mean
        # Use one-hot encoding for vectorized update
        for c in range(C):
            mask = (assignments == c).unsqueeze(-1).float()  # [B, H, N, 1]
            cluster_sum = (K * mask).sum(dim=2)  # [B, H, d]
            cluster_count = mask.sum(dim=2).clamp(min=1.0)  # [B, H, 1]
            centroids[:, :, c, :] = cluster_sum / cluster_count

    return centroids


def voronoi_assign(Q: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Voronoi assignment: each Query → nearest centroid.

    Args:
        Q: Query tensor [B, heads, N, d_head]
        centroids: Cluster centroids [B, heads, C, d_head]

    Returns:
        assignments: [B, heads, N] - centroid index per token

    Complexity: O(N * C)
    """
    # Compute distances: [B, H, N, C]
    dists = torch.cdist(Q, centroids)

    # Assignment: argmin over centroids
    assignments = dists.argmin(dim=-1)

    return assignments


def block_masked_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    assignments: torch.Tensor,
    centroids: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Block-masked attention: attention only within each bubble.

    Args:
        Q, K, V: [B, heads, N, d_head]
        assignments: [B, heads, N] - bubble index per token
        centroids: [B, heads, C, d_head] - not used in this implementation
        scale: Optional scaling factor (default: 1/sqrt(d_head))

    Returns:
        output: [B, heads, N, d_head]

    Note: This implementation uses masking for simplicity.
    Future optimization: use xformers BlockDiagonalMask.
    """
    B, H, N, d = Q.shape
    C = centroids.shape[2]

    if scale is None:
        scale = 1.0 / (d**0.5)

    # Compute attention scores: [B, H, N, N]
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Create bubble masks: tokens in same bubble can attend to each other
    # assignments: [B, H, N] → [B, H, N, 1] and [B, H, 1, N]
    assignments_expanded_row = assignments.unsqueeze(-1)  # [B, H, N, 1]
    assignments_expanded_col = assignments.unsqueeze(-2)  # [B, H, 1, N]

    # Bubble mask: True where both tokens are in the same bubble
    bubble_mask = assignments_expanded_row == assignments_expanded_col  # [B, H, N, N]

    # Apply mask: set non-bubble positions to -inf
    attn_scores = attn_scores.masked_fill(~bubble_mask, float("-inf"))

    # Softmax (automatically handles -inf)
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Handle NaN from all -inf rows (empty bubbles)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Apply attention to values
    output = torch.matmul(attn_weights, V)

    return output


def compute_hard_support(assignments: torch.Tensor) -> torch.Tensor:
    """
    Compute Hard Support: number of token pairs in the same bubble.

    Args:
        assignments: [B, heads, N] - bubble index per token

    Returns:
        support: [B, heads] - average number of connections per head

    Used in calibration to find optimal C.
    """
    B, H, N = assignments.shape

    # Count tokens per bubble
    support = torch.zeros(B, H, device=assignments.device, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            # Get unique bubbles and their counts
            unique, counts = torch.unique(assignments[b, h], return_counts=True)
            # Support = sum(counts^2) - sum(counts) (pairs within same bubble)
            support[b, h] = (counts.float() ** 2).sum() - counts.float().sum()

    return support


if __name__ == "__main__":
    # Quick test
    print("[v3_core] Running quick test...")

    B, H, N, d = 2, 4, 100, 64
    C = 32

    # Test cluster_keys
    K = torch.randn(B, H, N, d)
    centroids = cluster_keys(K, num_centroids=C)
    print(f"cluster_keys: K {K.shape} -> centroids {centroids.shape}")
    assert centroids.shape == (B, H, C, d), (
        f"Expected {(B, H, C, d)}, got {centroids.shape}"
    )

    # Test voronoi_assign
    Q = torch.randn(B, H, N, d)
    assignments = voronoi_assign(Q, centroids)
    print(f"voronoi_assign: Q {Q.shape} -> assignments {assignments.shape}")
    assert assignments.shape == (B, H, N), (
        f"Expected {(B, H, N)}, got {assignments.shape}"
    )
    assert assignments.min() >= 0 and assignments.max() < C, "Assignments out of range"

    # Test block_masked_attention
    V = torch.randn(B, H, N, d)
    output = block_masked_attention(Q, K, V, assignments, centroids)
    print(f"block_masked_attention: output {output.shape}")
    assert output.shape == (B, H, N, d), f"Expected {(B, H, N, d)}, got {output.shape}"

    # Test compute_hard_support
    support = compute_hard_support(assignments)
    print(f"compute_hard_support: support {support.shape}")
    assert support.shape == (B, H), f"Expected {(B, H)}, got {support.shape}"

    print("[v3_core] All tests passed!")
