"""Gumbel-Sinkhorn attention variant.

Based on Mena et al. 2018 (arXiv:1802.08665) and Tay et al. 2020 (arXiv:2002.11296).
Idea: Gumbel noise + Sinkhorn iterations produces near-permutation matrices
that are inherently peaked while being doubly-stochastic.

This addresses the "SIRI peakedness problem" by producing attention matrices
that are sparse-like (near permutations) while maintaining doubly-stochasticity.
"""
import torch
import numpy as np


def gumbel_sinkhorn_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    epsilon: float = 0.1,
    tau: float = 1.0,
    n_sinkhorn_iters: int = 5,
    n_gumbel_samples: int = 1,
    causal: bool = True,
) -> torch.Tensor:
    """Gumbel-Sinkhorn attention.

    Args:
        Q: queries [B, H, N, d]
        K: keys [B, H, N, d]
        V: values [B, H, N, d]
        epsilon: bandwidth for geometric cost
        tau: Gumbel temperature (lower = more peaked)
        n_sinkhorn_iters: number of Sinkhorn iterations
        n_gumbel_samples: number of Gumbel samples to average
        causal: apply causal mask

    Returns:
        output: [B, H, N, d]
    """
    B, H, N, d = Q.shape
    device = Q.device
    orig_dtype = Q.dtype

    # Work in float32 for numerical stability
    Q_f32 = Q.float()
    K_f32 = K.float()
    V_f32 = V.float()

    # Normalize Q, K for stable geometric cost
    Q_norm = Q_f32 / (Q_f32.norm(dim=-1, keepdim=True) + 1e-8)
    K_norm = K_f32 / (K_f32.norm(dim=-1, keepdim=True) + 1e-8)

    # Geometric cost C_ij = ||Q_i - K_j||^2
    # C = ||Q||^2 + ||K||^2 - 2*Q.K^T
    # Since Q,K are normalized, ||Q||^2 = ||K||^2 = 1
    # So C_ij = 2 - 2*Q.K^T = 2(1 - cos(Q, K))
    Q_sq = (Q_norm ** 2).sum(dim=-1, keepdim=True)  # [B, H, N, 1]
    K_sq = (K_norm ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)  # [B, H, 1, N]
    QK = torch.matmul(Q_norm, K_norm.transpose(-1, -2))  # [B, H, N, N]
    C = Q_sq + K_sq - 2 * QK  # [B, H, N, N], values in [0, 4]

    # Initialize log-S with Gumbel noise for sparsity
    # log_S = -C/epsilon + Gumbel(0, tau)
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(C) + 1e-20) + 1e-20
    ) * tau
    log_S = -C / epsilon + gumbel_noise

    # Causal mask
    if causal:
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        log_S.masked_fill_(mask, -1e9)

    # Sinkhorn iterations in log-domain
    log_S = _sinkhorn_log_domain(log_S, n_sinkhorn_iters)

    # Convert to attention weights
    attn = torch.exp(log_S)  # [B, H, N, N]

    # Apply to values
    out = torch.matmul(attn, V_f32)  # [B, H, N, d]
    return out.to(orig_dtype)


def _sinkhorn_log_domain(log_M: torch.Tensor, n_iters: int) -> torch.Tensor:
    """Sinkhorn normalization in log-domain.

    Numerically stable for small epsilon by working in log space.
    """
    log_M = log_M.clone()
    for _ in range(n_iters):
        # Row normalization (log-sum-exp)
        log_M = log_M - torch.logsumexp(log_M, dim=-1, keepdim=True)
        # Column normalization (log-sum-exp)
        log_M = log_M - torch.logsumexp(log_M, dim=-2, keepdim=True)
    return log_M


def sparse_sinkhorn_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    epsilon: float = 0.1,
    top_k: int = 16,
    n_sinkhorn_iters: int = 5,
    causal: bool = True,
) -> torch.Tensor:
    """Sparse-then-Sinkhorn attention.

    Based on Tay et al. 2020 (arXiv:2002.11296).
    Apply top-k sparse attention first to preserve peakedness,
    then Sinkhorn-balance the residual for doubly-stochastic regularization.
    """
    B, H, N, d = Q.shape
    device = Q.device
    orig_dtype = Q.dtype

    Q_f32 = Q.float()
    K_f32 = K.float()
    V_f32 = V.float()

    # Normalize
    Q_norm = Q_f32 / (Q_f32.norm(dim=-1, keepdim=True) + 1e-8)
    K_norm = K_f32 / (K_f32.norm(dim=-1, keepdim=True) + 1e-8)

    # Geometric cost
    Q_sq = (Q_norm ** 2).sum(dim=-1, keepdim=True)
    K_sq = (K_norm ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)
    QK = torch.matmul(Q_norm, K_norm.transpose(-1, -2))
    C = Q_sq + K_sq - 2 * QK

    # Causal mask: only attend to past
    if causal:
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        C.masked_fill_(mask, 1e9)  # large cost for future

    # Top-k: keep only k smallest costs (most similar)
    k = min(top_k, N)
    topk_costs, topk_indices = C.topk(k, dim=-1, largest=False)  # [B, H, N, k]

    # Build sparse log-S
    log_S = torch.full_like(C, -1e9)
    log_S.scatter_(-1, topk_indices, -topk_costs / epsilon)

    # Sinkhorn
    log_S = _sinkhorn_log_domain(log_S, n_sinkhorn_iters)

    # Apply
    attn = torch.exp(log_S)
    out = torch.matmul(attn, V_f32)
    return out.to(orig_dtype)


if __name__ == "__main__":
    # Quick self-test
    B, H, N, d = 1, 4, 32, 64
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    print("Testing Gumbel-Sinkhorn...")
    out = gumbel_sinkhorn_attention(Q, K, V, tau=0.5, n_sinkhorn_iters=5)
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

    print("\nTesting Sparse-Sinkhorn...")
    out = sparse_sinkhorn_attention(Q, K, V, top_k=8, n_sinkhorn_iters=5)
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

    # Verify sparsity: check how many non-zero entries
    Q_norm = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
    K_norm = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
    Q_sq = (Q_norm ** 2).sum(dim=-1, keepdim=True)
    K_sq = (K_norm ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)
    C = Q_sq + K_sq - 2 * torch.matmul(Q_norm, K_norm.transpose(-1, -2))
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    C.masked_fill_(mask, 1e9)
    k = 8
    topk_costs, _ = C.topk(k, dim=-1, largest=False)
    log_S = torch.full_like(C, -1e9)
    log_S.scatter_(-1, C.topk(k, dim=-1, largest=False)[1], -topk_costs / 0.1)
    from torch.nn.functional import normalize
    for _ in range(5):
        log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
        log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
    attn = torch.exp(log_S)
    nonzero_per_row = (attn > 1e-6).sum(dim=-1).float().mean().item()
    print(f"  Avg non-zero entries per row: {nonzero_per_row:.1f} / {N}")
    print(f"  Sparsity: {(1 - nonzero_per_row / N) * 100:.1f}%")
