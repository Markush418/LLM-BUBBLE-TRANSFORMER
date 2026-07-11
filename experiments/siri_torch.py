"""Pure PyTorch SIRI — zero NumPy bridge.

Rewrites siri_soft.py (NumPy) as native torch operations.
Expected speedup: 5-20x from eliminating CPU↔GPU copies alone.
"""
import torch
import torch.nn.functional as F
from typing import Optional


def _logsumexp_torch(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Numerically stable log-sum-exp, pure torch."""
    m, _ = x.max(dim=dim, keepdim=True)
    out = m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=True) + 1e-30)
    if not keepdim:
        out = out.squeeze(dim)
    return out


def sinkhorn_log_domain(
    log_S: torch.Tensor,
    n_iters: int = 5,
) -> torch.Tensor:
    """Sinkhorn-Knopp in log-domain, pure PyTorch.

    Args:
        log_S: [..., N, N] log-domain kernel (any batch dims)
        n_iters: number of Sinkhorn iterations

    Returns:
        A: [..., N, N] doubly-stochastic matrix
    """
    log_S = log_S.clone()
    for _ in range(n_iters):
        # Row normalization: u = -logsumexp(log_S, dim=-1)
        log_S = log_S - _logsumexp_torch(log_S, dim=-1, keepdim=True)
        # Column normalization: v = -logsumexp(log_S, dim=-2)
        log_S = log_S - _logsumexp_torch(log_S, dim=-2, keepdim=True)
    return torch.exp(log_S)


def siri_classical_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    epsilon: float = 0.1,
    tau_iters: int = 5,
    causal: bool = False,
) -> torch.Tensor:
    """Pure PyTorch SIRI classical attention.

    Args:
        Q: [B, H, N, D_h]
        K: [B, H, N, D_h]
        V: [B, H, N, D_h]
        epsilon: bandwidth
        tau_iters: Sinkhorn iterations
        causal: apply causal mask

    Returns:
        output: [B, H, N, D_h]
    """
    B, H, N, D_h = Q.shape

    # Geometric cost: C = ||Q_i - K_j||^2
    Q_sq = (Q * Q).sum(dim=-1, keepdim=True)     # [B, H, N, 1]
    K_sq = (K * K).sum(dim=-1, keepdim=True)      # [B, H, N, 1]
    C = Q_sq + K_sq.transpose(-2, -1) - 2.0 * torch.matmul(Q, K.transpose(-2, -1))
    C = C.clamp(min=0.0)

    # log_S = -C / epsilon
    log_S = -C / epsilon

    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        log_S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -1e9)

    # Sinkhorn
    A = sinkhorn_log_domain(log_S, n_iters=tau_iters)

    return torch.matmul(A, V)


def siri_soft_blend_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    alpha: float = 0.7,
    epsilon: float = 0.1,
    tau_iters: int = 5,
    causal: bool = False,
) -> torch.Tensor:
    """Pure PyTorch SIRI-Soft: A = (1-alpha)*softmax + alpha*siri.

    Args:
        Q: [B, H, N, D_h]
        K: [B, H, N, D_h]
        V: [B, H, N, D_h]
        alpha: blend coefficient (0=pure softmax, 1=pure SIRI)
        epsilon: bandwidth
        tau_iters: Sinkhorn iterations
        causal: apply causal mask

    Returns:
        output: [B, H, N, D_h]
    """
    B, H, N, D_h = Q.shape

    # Geometric cost
    Q_sq = (Q * Q).sum(dim=-1, keepdim=True)
    K_sq = (K * K).sum(dim=-1, keepdim=True)
    C = Q_sq + K_sq.transpose(-2, -1) - 2.0 * torch.matmul(Q, K.transpose(-2, -1))
    C = C.clamp(min=0.0)

    log_S = -C / epsilon

    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        log_S.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -1e9)

    # SIRI part
    A_siri = sinkhorn_log_domain(log_S, n_iters=tau_iters)

    # Softmax part (use geometric cost as scores)
    scores = -C  # negative cost as similarity scores
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    A_softmax = F.softmax(scores, dim=-1)

    # Blend
    A = (1.0 - alpha) * A_softmax + alpha * A_siri

    return torch.matmul(A, V)


def gumbel_sinkhorn_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    epsilon: float = 0.1,
    tau: float = 0.1,
    n_sinkhorn_iters: int = 5,
    causal: bool = False,
) -> torch.Tensor:
    """Pure PyTorch Gumbel-Sinkhorn attention.

    Args:
        Q: [B, H, N, D_h]
        K: [B, H, N, D_h]
        V: [B, H, N, D_h]
        epsilon: bandwidth
        tau: Gumbel temperature (lower = more peaked)
        n_sinkhorn_iters: Sinkhorn iterations
        causal: apply causal mask

    Returns:
        output: [B, H, N, D_h]
    """
    B, H, N, D_h = Q.shape
    device = Q.device
    orig_dtype = Q.dtype

    # Work in float32
    Q_f32 = Q.float()
    K_f32 = K.float()
    V_f32 = V.float()

    # Normalize for geometric cost
    Q_norm = Q_f32 / (Q_f32.norm(dim=-1, keepdim=True) + 1e-8)
    K_norm = K_f32 / (K_f32.norm(dim=-1, keepdim=True) + 1e-8)

    Q_sq = (Q_norm ** 2).sum(dim=-1, keepdim=True)
    K_sq = (K_norm ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)
    QK = torch.matmul(Q_norm, K_norm.transpose(-1, -2))
    C = Q_sq + K_sq - 2 * QK

    # Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(C) + 1e-20) + 1e-20) * tau
    log_S = -C / epsilon + gumbel_noise

    if causal:
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        log_S.masked_fill_(mask, -1e9)

    # Sinkhorn
    log_S_cloned = log_S.clone()
    for _ in range(n_sinkhorn_iters):
        log_S_cloned = log_S_cloned - _logsumexp_torch(log_S_cloned, dim=-1, keepdim=True)
        log_S_cloned = log_S_cloned - _logsumexp_torch(log_S_cloned, dim=-2, keepdim=True)

    attn = torch.exp(log_S_cloned)
    out = torch.matmul(attn, V_f32)
    return out.to(orig_dtype)


if __name__ == "__main__":
    print("[SIRI Pure PyTorch] Smoke test...")
    B, H, N, D_h = 1, 4, 64, 64
    Q = torch.randn(B, H, N, D_h, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D_h, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D_h, device="cuda", dtype=torch.float16)

    out = siri_classical_torch(Q, K, V, epsilon=0.1, tau_iters=5)
    print(f"  Classical: {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")

    out = siri_soft_blend_torch(Q, K, V, alpha=0.7, epsilon=0.1, tau_iters=5)
    print(f"  Soft blend: {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")

    out = gumbel_sinkhorn_torch(Q, K, V, epsilon=0.1, tau=0.1, n_sinkhorn_iters=5)
    print(f"  Gumbel-Sinkhorn: {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")

    print("[SIRI Pure PyTorch] Passed!")
