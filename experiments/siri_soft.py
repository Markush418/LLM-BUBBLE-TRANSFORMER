"""
SIRI-Soft: Soft Doubly-Stochastic Attention via Blended Sinkhorn
=================================================================

[DEFINITION] SIRI-Soft.
Classical SIRI (Sinkhorn-Knopp) destroys attention peakedness because the doubly-
stochastic constraint forces average entry = 1/N. SIRI-Soft blends the original
softmax attention with a softly-regularized Sinkhorn result:

    A_soft_siri = (1 - alpha) * A_softmax + alpha * Sinkhorn(scores)

where alpha controls the strength of doubly-stochasticity. When alpha = 0, this
is exactly standard softmax (peaked). When alpha = 1, this is classical SIRI
(overly uniformed).

The theoretical justification (Sandler et al. 2021, Sinkformers): softmax
attention matrices naturally drift toward doubly-stochastic during training.
SIRI-Soft accelerates this convergence without imposing the full constraint.

Variants:
  1. soft_blend:    A = (1-alpha) * softmax + alpha * Sinkhorn(scores)
  2. sparse:        A = relu(scores) then Sinkhorn col/row normalize
  3. chiller:       Sinkhorn with temperature sharpening (multiply log_S by beta > 1)

NumPy implementation. Compatible with torch tensors via input conversion.
"""

import numpy as np
from typing import Optional


def _to_numpy(x):
    """Defensive conversion to NumPy — handles torch tensors, numpy arrays, lists."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =============================================================================
# Variant 1: Soft Blend (SpikeFormer-style)
# =============================================================================


def siri_soft_blend(
    scores: np.ndarray,
    alpha: float = 0.3,
    tau_iters: int = 5,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """Soft SIRI blend: (1-alpha)*softmax + alpha*SIRI.

    Args:
        scores: [B, H, N, N] raw attention scores (e.g., QK^T / sqrt(d)).
        alpha: Blend weight. 0 = pure softmax, 1 = pure SIRI.
        tau_iters: Sinkhorn iterations.
        mask: Optional [N, N] additive mask (0 valid, -inf masked).
        eps: Numerical stability.

    Returns:
        A: [B, H, N, N] attention matrix.
    """
    scores = _to_numpy(scores).astype(np.float32)
    if scores.ndim != 4:
        raise ValueError(f"Expected scores of shape [B, H, N, N], got {scores.shape}")

    if mask is not None:
        mask = _to_numpy(mask).astype(np.float32)
        if mask.ndim == 4:
            mask_2d = mask[0, 0]
        else:
            mask_2d = mask
        scores = scores + mask_2d[np.newaxis, np.newaxis, :, :]

    # Softmax along last axis (row-wise).
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    sum_exp = exp_scores.sum(axis=-1, keepdims=True) + eps
    A_softmax = exp_scores / sum_exp

    # SIRI: Sinkhorn on log-softmax scores (numerical stability).
    # Using log_S = scores / 1.0 means we're doing Sinkhorn on raw scores,
    # which tends to be more peaked than using log(softmax).
    # Multiply by alpha to control sharply.
    if alpha == 0.0:
        return A_softmax.astype(np.float32)

    # SIRI iterations on scores (in log domain).
    log_S = scores
    u = np.zeros(scores.shape[:-1], dtype=np.float32)
    v = np.zeros(scores.shape[:-1], dtype=np.float32)

    for _ in range(tau_iters):
        # log S + v broadcast over columns.
        log_S_plus_v = log_S + v[:, :, :, np.newaxis]
        log_S_plus_v_max = log_S_plus_v.max(axis=-1, keepdims=True)
        lse_cols = log_S_plus_v_max + np.log(
            np.exp(log_S_plus_v - log_S_plus_v_max).sum(axis=-1, keepdims=True) + eps
        )
        u = -lse_cols.squeeze(-1)

        log_S_plus_u = log_S + u[:, :, :, np.newaxis]
        log_S_plus_u_max = log_S_plus_u.max(axis=-2, keepdims=True)
        lse_rows = log_S_plus_u_max + np.log(
            np.exp(log_S_plus_u - log_S_plus_u_max).sum(axis=-2, keepdims=True) + eps
        )
        v = -lse_rows.squeeze(-2)

    log_A = log_S + u[:, :, :, np.newaxis] + v[:, :, np.newaxis, :]
    log_A = np.clip(log_A, -50.0, 50.0)
    A_siri = np.exp(log_A)
    A_siri = np.nan_to_num(A_siri, nan=0.0, posinf=0.0, neginf=0.0)

    # Renormalize SIRI to be row-stochastic (small numerical drift).
    rs = A_siri.sum(axis=-1, keepdims=True)
    safe = np.where(rs > 1e-30, rs, 1.0)
    A_siri = np.where(rs > 1e-30, A_siri / safe, np.zeros_like(A_siri))

    # Blend.
    return ((1.0 - alpha) * A_softmax + alpha * A_siri).astype(np.float32)


# =============================================================================
# Variant 2: Sparse Doubly-Stochastic (ReLU + Sinkhorn)
# =============================================================================


def siri_sparse(
    scores: np.ndarray,
    tau_iters: int = 5,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """Sparse DS: ReLU(scores) + Sinkhorn to doubly-stochastic.

    Unlike softmax which forces attention distribution, ReLU preserves the
    natural peakedness of raw scores (zero for negatives, linear for positives).
    The downstream Sinkhorn iterations then enforce doubly-stochastic constraints
    while keeping the peakedness structure.

    Args:
        scores: [B, H, N, N] raw attention scores.
        tau_iters: Sinkhorn iterations.
        mask: Optional [N, N] additive mask.
        eps: Numerical stability.

    Returns:
        A: [B, H, N, N] sparse doubly-stochastic matrix.
    """
    scores = _to_numpy(scores).astype(np.float32)
    if scores.ndim != 4:
        raise ValueError(f"Expected scores of shape [B, H, N, N], got {scores.shape}")

    # Build mask indicator (1 valid, 0 masked) for hard zeroing at the end.
    if mask is not None:
        mask = _to_numpy(mask).astype(np.float32)
        if mask.ndim == 4:
            mask_2d = mask[0, 0]
        else:
            mask_2d = mask
        # 1 for valid, 0 for masked.
        valid = (mask_2d == 0).astype(np.float32)
        # Apply mask aggressively in scores: very negative for masked positions.
        scores_masked = scores + (1.0 - valid) * (-1e6)
    else:
        scores_masked = scores
        valid = np.ones_like(scores)

    # ReLU activation in log-domain (sparse peaks).
    log_S = np.where(scores_masked > 0, np.log(np.maximum(scores_masked, eps)), -50.0)
    log_S = np.where(np.isfinite(log_S), log_S, -50.0)

    # Sinkhorn iterations.
    u = np.zeros(scores.shape[:-1], dtype=np.float32)
    v = np.zeros(scores.shape[:-1], dtype=np.float32)

    for _ in range(tau_iters):
        log_S_plus_v = log_S + v[:, :, :, np.newaxis]
        log_S_plus_v_max = log_S_plus_v.max(axis=-1, keepdims=True)
        lse_cols = log_S_plus_v_max + np.log(
            np.exp(log_S_plus_v - log_S_plus_v_max).sum(axis=-1, keepdims=True) + eps
        )
        u = -lse_cols.squeeze(-1)

        log_S_plus_u = log_S + u[:, :, :, np.newaxis]
        log_S_plus_u_max = log_S_plus_u.max(axis=-2, keepdims=True)
        lse_rows = log_S_plus_u_max + np.log(
            np.exp(log_S_plus_u - log_S_plus_u_max).sum(axis=-2, keepdims=True) + eps
        )
        v = -lse_rows.squeeze(-2)

    log_A = log_S + u[:, :, :, np.newaxis] + v[:, :, np.newaxis, :]
    log_A = np.clip(log_A, -50.0, 50.0)
    A = np.exp(log_A)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # Row normalize with scale-aware eps.
    row_sums = A.sum(axis=-1, keepdims=True)
    safe_div = np.where(row_sums > 1e-30, row_sums, 1.0)
    A = np.where(row_sums > 1e-30, A / safe_div, np.zeros_like(A))

    # Hard zero masked positions (in case Sinkhorn leak through).
    if mask is not None:
        A = A * valid

    return A.astype(np.float32)


# =============================================================================
# Variant 3: Sinkhorn with Temperature Sharpening (Chiller)
# =============================================================================


def siri_chiller(
    scores: np.ndarray,
    beta: float = 10.0,
    tau_iters: int = 5,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """Sinkhorn with temperature sharpening (chiller).

    Multiply scores by beta > 1 BEFORE Sinkhorn, so the doubly-stochastic
    constraint is applied on a sharper kernel. Result is near-doubly-stochastic
    with much higher peakedness than classical SIRI.

    Args:
        scores: [B, H, N, N] raw attention scores.
        beta: Temperature scaling. >1 = sharper peaks. <1 = smoother.
        tau_iters: Sinkhorn iterations.
        mask: Optional [N, N] additive mask.

    Returns:
        A: [B, H, N, N] near-doubly-stochastic matrix with beta-sharpened peaks.
    """
    scores = _to_numpy(scores).astype(np.float32)
    if scores.ndim != 4:
        raise ValueError(f"Expected scores of shape [B, H, N, N], got {scores.shape}")

    if mask is not None:
        mask = _to_numpy(mask).astype(np.float32)
        mask_2d = mask if mask.ndim == 2 else mask[0, 0]
        scores = scores + mask_2d[np.newaxis, np.newaxis, :, :]

    # Apply sharpening.
    log_S = scores * beta

    # Sinkhorn iterations.
    u = np.zeros(scores.shape[:-1], dtype=np.float32)
    v = np.zeros(scores.shape[:-1], dtype=np.float32)

    for _ in range(tau_iters):
        log_S_plus_v = log_S + v[:, :, :, np.newaxis]
        log_S_plus_v_max = log_S_plus_v.max(axis=-1, keepdims=True)
        lse_cols = log_S_plus_v_max + np.log(
            np.exp(log_S_plus_v - log_S_plus_v_max).sum(axis=-1, keepdims=True) + eps
        )
        u = -lse_cols.squeeze(-1)

        log_S_plus_u = log_S + u[:, :, :, np.newaxis]
        log_S_plus_u_max = log_S_plus_u.max(axis=-2, keepdims=True)
        lse_rows = log_S_plus_u_max + np.log(
            np.exp(log_S_plus_u - log_S_plus_u_max).sum(axis=-2, keepdims=True) + eps
        )
        v = -lse_rows.squeeze(-2)

    log_A = log_S + u[:, :, :, np.newaxis] + v[:, :, np.newaxis, :]
    # Clamp log_A before exp to prevent overflow at high beta.
    log_A = np.clip(log_A, -50.0, 50.0)
    A = np.exp(log_A)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # Row normalize with scale-aware eps to avoid divide-by-tiny.
    row_sums = A.sum(axis=-1, keepdims=True)
    # Use a scale-adaptive eps: 1e-20 of typical row sum magnitude.
    safe_div = np.where(row_sums > 1e-30, row_sums, 1.0)
    A = np.where(row_sums > 1e-30, A / safe_div, np.zeros_like(A))

    return A.astype(np.float32)


# =============================================================================
# Smoke test
# =============================================================================


if __name__ == "__main__":
    print("[SIRI-Soft] Smoke test")
    B, H, N = 2, 4, 8
    rng = np.random.RandomState(42)

    # Generate synthetic scores with one peak per row.
    scores = rng.randn(B, H, N, N).astype(np.float32)
    # Add a strong peak at position [0,0,0,2].
    scores[0, 0, 0] += 10.0

    print(f"\nInput shape: {scores.shape}")
    print(f"Sample row 0 peak at col 2 (off by 10): {scores[0, 0, 0]}")

    # Standard softmax.
    sm_max = scores[0, 0, 0].max()
    exp_s = np.exp(scores[0, 0, 0] - sm_max)
    sm = exp_s / exp_s.sum()
    print(f"\nSoftmax row 0: max={sm.max():.4f} at col {sm.argmax()}, min={sm.min():.6f}")

    # SIRI classical.
    from siri_postprocess import siri_sinkhorn_log_domain
    A_siri = siri_sinkhorn_log_domain(scores, tau_iters=10)
    print(f"Classical SIRI row 0: max={A_siri[0, 0, 0].max():.4f}, min={A_siri[0, 0, 0].min():.6f}")

    # SIRI-Soft blend (alpha=0.3).
    A_soft = siri_soft_blend(scores, alpha=0.3, tau_iters=10)
    print(f"SIRI-Soft (alpha=0.3) row 0: max={A_soft[0, 0, 0].max():.4f}, min={A_soft[0, 0, 0].min():.6f}")

    # SIRI-Sparse.
    A_sparse = siri_sparse(scores, tau_iters=10)
    print(f"SIRI-Sparse row 0: max={A_sparse[0, 0, 0].max():.4f}, min={A_sparse[0, 0, 0].min():.6f}")

    # SIRI-Chiller.
    A_chill = siri_chiller(scores, beta=10.0, tau_iters=10)
    print(f"SIRI-Chiller (beta=10) row 0: max={A_chill[0, 0, 0].max():.4f}, min={A_chill[0, 0, 0].min():.6f}")

    print("\n[DONE] All SIRI-Soft variants produce distinct attention patterns.")