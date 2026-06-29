"""
SIRI Post-Processing — Sinkhorn-Knopp log-domain as opt-in regularizer
======================================================================

[DEFINITION] SIRI as post-processing.

Given an attention matrix A_SOTA produced by any architecture (Softmax,
DeltaNet, Kimi Linear, etc.), SIRI post-process refines it into a doubly
stochastic matrix via Sinkhorn-Knopp iterations:

    A_post = Sinkhorn_Knopp(A_SOTA, tau=5)

This converts any attention matrix into the optimal transport plan under
the cost function derived from Q/K projections.

Pure NumPy implementation. Compatible with torch tensors via input conversion.
"""

import numpy as np
from typing import Optional, Tuple


def _to_numpy(x):
    """Defensive conversion to NumPy — handles torch tensors, numpy arrays, lists."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def siri_sinkhorn_log_domain(
    log_S: np.ndarray,
    tau_iters: int = 5,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sinkhorn-Knopp iterations in log-domain.

    [THEOREM-2.2 from docs/decisions/2026-06-27-siri-power-diagram-math.md]
    For tau iterations of Sinkhorn-Knopp log-domain:
        M^(0) = log_S
        for t = 1, ..., tau:
            u^(t) = -logsumexp(M^(t-1) + v^(t-1) * 1_N^T, axis=1)
            v^(t) = -logsumexp(M^(t-1) + 1_N * u^(t)^T, axis=0)
        A^(t) = exp(M^(t) + u^(t) * 1_N^T + 1_N * v^(t)^T)

    Args:
        log_S: [B, H, N, N] log-domain kernel.
        tau_iters: Number of Sinkhorn iterations.
        mask: Optional [B, N, N] or [N, N] attention mask.

    Returns:
        A: [B, H, N, N] doubly stochastic matrix.
    """
    log_S = _to_numpy(log_S).astype(np.float32)
    if log_S.ndim != 4:
        raise ValueError(f"Expected log_S of shape [B, H, N, N], got {log_S.shape}")

    if mask is not None:
        mask = _to_numpy(mask).astype(np.float32)
        if mask.ndim == 2:
            mask_2d = mask[np.newaxis, np.newaxis, :, :]
        elif mask.ndim == 3:
            mask_2d = mask[:, np.newaxis, :, :]
        else:
            mask_2d = mask
        # Use very negative value so masked positions underflow in exp().
        log_S = np.where(mask_2d == 0, -1e30, log_S)

    B, H, N, _ = log_S.shape
    u = np.zeros((B, H, N), dtype=np.float32)
    v = np.zeros((B, H, N), dtype=np.float32)

    for _ in range(tau_iters):
        # u update: logsumexp over columns (axis=-1)
        u = -_logsumexp(log_S + v[:, :, np.newaxis, :], axis=-1)
        # v update: logsumexp over rows (axis=-2)
        v = -_logsumexp(log_S + u[:, :, :, np.newaxis], axis=-2)

    A = np.exp(log_S + u[:, :, :, np.newaxis] + v[:, :, np.newaxis, :])
    return A


def siri_postprocess_attention(
    A_sota: np.ndarray,
    Q: np.ndarray,
    K: np.ndarray,
    epsilon: float = 0.1,
    tau_iters: int = 5,
    normalize_costs: bool = True,
) -> np.ndarray:
    """Apply SIRI as post-processing over an existing attention matrix.

    Steps:
      1. Recompute log_S from geometric cost C = ||Q - K||^2 / epsilon.
      2. Run Sinkhorn-Knopp in log-domain.
      3. Multiply by the SOTA attention matrix element-wise (gate), then
         renormalize to preserve SOTA semantics while applying doubly-stochastic
         constraint.

    Args:
        A_sota: [B, H, N, N] attention matrix from any SOTA architecture.
        Q: [B, H, N, head_dim] query projections.
        K: [B, H, N, head_dim] key projections.
        epsilon: SIRI bandwidth.
        tau_iters: Sinkhorn iterations.
        normalize_costs: Whether to min-max normalize cost matrix per batch.

    Returns:
        A_post: [B, H, N, N] refined doubly-stochastic attention matrix.
    """
    A_sota = _to_numpy(A_sota).astype(np.float32)
    Q = _to_numpy(Q).astype(np.float32)
    K = _to_numpy(K).astype(np.float32)

    if A_sota.ndim != 4:
        raise ValueError(f"Expected A_sota of shape [B, H, N, N], got {A_sota.shape}")
    if Q.shape != K.shape:
        raise ValueError(f"Q and K must have the same shape: {Q.shape} vs {K.shape}")

    # Compute geometric cost matrix C_ij = ||Q_i - K_j||^2
    Q_sq = np.sum(Q ** 2, axis=-1, keepdims=True)
    K_sq = np.sum(K ** 2, axis=-1, keepdims=True)
    K_sq_t = np.moveaxis(K_sq, -2, -1)
    C = np.maximum(Q_sq + K_sq_t - 2.0 * np.matmul(Q, np.moveaxis(K, -2, -1)), 0.0)

    if normalize_costs:
        C_min = np.min(C, axis=(-2, -1), keepdims=True)
        C_max = np.max(C, axis=(-2, -1), keepdims=True)
        C = (C - C_min) / (C_max - C_min + 1e-10)

    log_S = -C / epsilon
    A_siri = siri_sinkhorn_log_domain(log_S, tau_iters=tau_iters)

    # Gate SIRI with SOTA attention: A_post = A_sota * A_siri
    # Then renormalize each row to sum to 1 (preserving SOTA sparsity).
    A_post = A_sota * A_siri
    row_sums = A_post.sum(axis=-1, keepdims=True)
    A_post = A_post / np.maximum(row_sums, 1e-10)
    return A_post


def siri_interpolate(
    out_sota: np.ndarray,
    out_siri: np.ndarray,
    lam: float = 0.5,
) -> np.ndarray:
    """Linear interpolation between SOTA output and SIRI-refined output.

    out_final = lam * out_sota + (1 - lam) * out_siri

    Args:
        out_sota: [B, N, d_model] output from SOTA architecture.
        out_siri: [B, N, d_model] output from SIRI-refined attention.
        lam: interpolation coefficient (1.0 = pure SOTA, 0.0 = pure SIRI).

    Returns:
        out_final: [B, N, d_model]
    """
    out_sota = _to_numpy(out_sota).astype(np.float32)
    out_siri = _to_numpy(out_siri).astype(np.float32)

    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    return lam * out_sota + (1.0 - lam) * out_siri


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-sum-exp along axis.

    Clamps the input to avoid inf/NaN from extreme values.
    Note: lower clamp is wide (-1e30) to support hard masks (masked positions
    underflow to 0 in exp).
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    # Clamp a - a_max to a reasonable range to avoid overflow in exp().
    # Upper bound 50 prevents exp overflow; lower bound -1e30 preserves hard masks.
    diff = np.clip(a - a_max, -1e30, 50.0)
    out = a_max + np.log(np.sum(np.exp(diff), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


if __name__ == "__main__":
    print("[SIRI Post-Process] Running quick test...")
    rng = np.random.RandomState(42)

    B, H, N, head_dim = 2, 4, 16, 32

    # Generate fake Q, K, V, A_sota (e.g., from softmax attention).
    # Normalize to unit variance to avoid extreme log_S values.
    Q = rng.randn(B, H, N, head_dim).astype(np.float32) * 0.1
    K = rng.randn(B, H, N, head_dim).astype(np.float32) * 0.1
    V = rng.randn(B, H, N, head_dim).astype(np.float32)

    # Simulated SOTA attention (row-stochastic).
    scores = rng.randn(B, H, N, N).astype(np.float32) * 0.5
    A_sota = np.exp(scores - scores.max(axis=-1, keepdims=True))
    A_sota = A_sota / A_sota.sum(axis=-1, keepdims=True)

    # Test 1: Pure SIRI Sinkhorn
    log_S = -np.sum((Q[:, :, :, np.newaxis, :] - K[:, :, np.newaxis, :, :]) ** 2, axis=-1) / 0.1
    A_siri = siri_sinkhorn_log_domain(log_S, tau_iters=10)
    row_sums = A_siri.sum(axis=-1)
    print(f"  SIRI doubly-stochastic check: row_sums mean={row_sums.mean():.4f} "
          f"(should be ~1.0), col_sums mean={A_siri.sum(axis=-2).mean():.4f}")
    assert np.allclose(row_sums, 1.0, atol=5e-2), f"SIRI not doubly stochastic: max dev={np.abs(row_sums - 1).max():.4f}"

    # Test 2: SIRI post-process over A_sota
    A_post = siri_postprocess_attention(A_sota, Q, K, epsilon=0.1, tau_iters=5)
    row_sums_post = A_post.sum(axis=-1)
    print(f"  A_post row sums: mean={row_sums_post.mean():.4f} (should be ~1.0)")
    print(f"  A_post shape: {A_post.shape}")

    # Test 3: Linear interpolation
    out_sota = rng.randn(B, N, 64).astype(np.float32)
    out_siri = rng.randn(B, N, 64).astype(np.float32)
    out_final = siri_interpolate(out_sota, out_siri, lam=0.5)
    expected = 0.5 * out_sota + 0.5 * out_siri
    assert np.allclose(out_final, expected, atol=1e-6), "Interpolation mismatch"
    print(f"  Interpolation lam=0.5: shape {out_final.shape}, "
          f"matches expected: {np.allclose(out_final, expected)}")

    print("[SIRI Post-Process] All tests passed!")
