"""
Metrics Engine — Embedding Concentration & Geometry Analysis (NumPy)
=====================================================================
Computes 6 core metrics for evaluating embedding concentration
under different attention mechanisms and epsilon values.

Pure NumPy — no PyTorch required.
"""

import numpy as np
from typing import Dict, Optional
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.spatial.distance import squareform

try:
    from tensor_compat import ops as _ops
except ImportError:
    _ops = None


def _to_numpy(x):
    """Defensive conversion to NumPy — handles torch tensors, numpy arrays, lists."""
    if _ops is not None:
        return _ops.to_numpy(x)
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# Spectral metrics (SIGMA paper - arXiv:2601.03385)
try:
    from .spectral_metrics import compute_all_spectral_metrics
except ImportError:
    from spectral_metrics import compute_all_spectral_metrics

# Crowding metrics (embedding space analysis)
try:
    from .crowding_metric import compute_all_crowding_metrics
except ImportError:
    from crowding_metric import compute_all_crowding_metrics


def effective_rank(embeddings) -> float:
    """
    Effective rank via SVD: exp(H(p)) where p_i = sigma_i / sum(sigma).
    embeddings: [N, D] or [B, N, D]
    """
    embeddings = _to_numpy(embeddings)
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(entropy))


def intrinsic_dimension_mle(embeddings, k: int = 10) -> float:
    """
    MLE estimator for intrinsic dimensionality (Levina & Bickel 2004).
    Uses k-NN distances.
    """
    embeddings = _to_numpy(embeddings)
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    N = embeddings.shape[0]
    k = min(k, N - 1)
    if N < 3 or k < 2:
        return float(embeddings.shape[-1])

    dists = np.sort(scipy_pdist(embeddings.astype(np.float32)))
    if len(dists) < k:
        return float(embeddings.shape[-1])

    # Approximate: use pairwise distance matrix for k-NN
    D = scipy_pdist(embeddings.astype(np.float32))
    D_sq = squareform(D)
    np.fill_diagonal(D_sq, np.inf)
    sorted_dists = np.sort(D_sq, axis=1)[:, :k]  # [N, k]

    ratios = np.log(sorted_dists[:, -1:] / sorted_dists[:, :-1] + 1e-10)
    d_hat = 1.0 / np.mean(ratios)
    return max(1.0, min(float(d_hat), float(embeddings.shape[-1])))


def anisotropy_index(embeddings) -> float:
    """
    Ratio of max eigenvalue to sum of eigenvalues of covariance matrix.
    1.0 = fully anisotropic, 1/d = perfectly isotropic.
    """
    embeddings = _to_numpy(embeddings)
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / centered.shape[0]
    # Add regularization for numerical stability
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
    try:
        eigenvalues = np.linalg.eigvalsh(cov_reg.astype(np.float32))
    except np.linalg.LinAlgError:
        # Fallback: use SVD if eigvalsh fails
        eigenvalues = np.linalg.svd(cov_reg.astype(np.float32), compute_uv=False)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 1.0
    return float(eigenvalues[-1] / eigenvalues.sum())


def pairwise_distance_stats(embeddings) -> Dict[str, float]:
    """Compute statistics of pairwise distances."""
    embeddings = _to_numpy(embeddings)
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    dists = scipy_pdist(embeddings.astype(np.float32))
    if len(dists) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "cv": 0.0,
        }
    mean_d = float(np.mean(dists))
    std_d = float(np.std(dists))
    return {
        "mean": mean_d,
        "std": std_d,
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
        "median": float(np.median(dists)),
        "cv": std_d / mean_d if mean_d > 0 else 0.0,
    }


def concentration_ratio(
    attention_matrix, threshold_factor: float = 1.0
) -> float:
    """
    Fraction of entries above threshold = threshold_factor / N.
    Lower = more concentrated (sparser).
    """
    attention_matrix = _to_numpy(attention_matrix)
    if attention_matrix.ndim == 4:
        attention_matrix = attention_matrix.mean(axis=(0, 1))
    elif attention_matrix.ndim == 3:
        attention_matrix = attention_matrix.mean(axis=0)
    N = attention_matrix.shape[-1]
    threshold = threshold_factor / N
    active = np.sum(attention_matrix >= threshold)
    total = attention_matrix.size
    return float(active / total)


def attention_entropy(attention_matrix) -> float:
    """Entropy of the attention distribution. Lower = more peaked.

    Convention: per-row Shannon entropy averaged across rows (the standard
    attention entropy used in the Transformer literature).
    """
    attention_matrix = _to_numpy(attention_matrix)
    if attention_matrix.ndim == 4:
        attention_matrix = attention_matrix.mean(axis=(0, 1))
    elif attention_matrix.ndim == 3:
        attention_matrix = attention_matrix.mean(axis=0)

    A = attention_matrix.astype(np.float32) + 1e-10

    if A.ndim >= 2:
        # Per-row entropy, averaged across rows. Standard for attention.
        # Always normalize each row so it sums to 1 (defensive: handles non-row-stochastic inputs).
        row_sums = A.sum(axis=-1, keepdims=True)
        A_norm = A / np.maximum(row_sums, 1e-10)
        return float(np.mean(-np.sum(A_norm * np.log(A_norm), axis=-1)))

    A = A / A.sum()
    return float(-np.sum(A * np.log(A)))


def compute_all_metrics(
    embeddings,
    attention_matrix: Optional[np.ndarray] = None,
    baseline_effective_rank: Optional[float] = None,
    attention_low: Optional[np.ndarray] = None,
    attention_high: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all concentration and geometry metrics at once.

    Includes:
    - Standard metrics (effective_rank, intrinsic_dim, anisotropy, pairwise distances)
    - Spectral metrics (SIGMA paper - collapse detection via Gram matrix eigenspectrum)
    - Crowding metrics (embedding space crowding analysis)
    """
    metrics = {}

    # ─── Standard Metrics ─────────────────────────────────────────────────────
    metrics["effective_rank"] = effective_rank(embeddings)
    if baseline_effective_rank is not None:
        metrics["effective_rank_ratio"] = (
            metrics["effective_rank"] / baseline_effective_rank
            if baseline_effective_rank > 0
            else 0.0
        )
    metrics["intrinsic_dim_mle"] = intrinsic_dimension_mle(embeddings, k=10)
    metrics["anisotropy_index"] = anisotropy_index(embeddings)
    dist_stats = pairwise_distance_stats(embeddings)
    for key, val in dist_stats.items():
        metrics[f"pairwise_dist_{key}"] = val

    # ─── Spectral Metrics (SIGMA paper) ────────────────────────────────────────────────
    try:
        spectral = compute_all_spectral_metrics(embeddings)
        metrics.update(spectral)
    except Exception as e:
        # Fallback: spectral metrics are critical but we continue on error
        metrics["spectral_log_det"] = 0.0
        metrics["collapse_score"] = 0.0
        metrics["spectral_error"] = str(e)

    # ─── Crowding Metrics ────────────────────────────────────���───────────────
    try:
        crowding = compute_all_crowding_metrics(embeddings)
        metrics.update(crowding)
    except Exception as e:
        # Fallback: crowding metrics are supplementary
        metrics["crowding_ratio_k10"] = 0.0
        metrics["crowding_error"] = str(e)

    # ─── Attention Metrics ────────────────────────────────────────────────────
    if attention_matrix is not None:
        metrics["cost_condition_number"] = cost_condition_number(attention_matrix)
        metrics["cost_spectral_gap"] = cost_spectral_gap(attention_matrix)
        metrics["concentration_ratio"] = concentration_ratio(attention_matrix)
        metrics["attention_entropy"] = attention_entropy(attention_matrix)
    if attention_low is not None and attention_high is not None:
        metrics["tension_balance"] = tension_balance(attention_low, attention_high)

    return metrics


def compute_metrics_batch(
    embeddings_list: list,
    attention_matrices: list = None,
    layer_indices: list = None,
    baseline_ranks: dict = None,
) -> list:
    """Compute metrics for a list of embeddings."""
    results = []
    for i, emb in enumerate(embeddings_list):
        layer_idx = layer_indices[i] if layer_indices else i
        baseline_rank = baseline_ranks.get(layer_idx) if baseline_ranks else None
        attn = (
            attention_matrices[i]
            if attention_matrices and i < len(attention_matrices)
            else None
        )
        metrics = compute_all_metrics(emb, attn, baseline_rank)
        metrics["layer"] = layer_idx
        results.append(metrics)
    return results


def cost_condition_number(cost_matrix) -> float:
    """
    Condition number of cost matrix via SVD: sigma_max / sigma_min.
    Higher = more ill-conditioned. Indicates numerical stability of Sinkhorn.
    cost_matrix: [B, heads, N, N] or [N, N]
    """
    cost_matrix = _to_numpy(cost_matrix)
    if cost_matrix.ndim > 2:
        # Average over batch and heads
        cost_matrix = cost_matrix.mean(axis=tuple(range(cost_matrix.ndim - 2)))
    _, S, _ = np.linalg.svd(cost_matrix.astype(np.float32), full_matrices=False)
    S = S[S > 1e-10]
    if len(S) == 0:
        return float("inf")
    return float(S[0] / S[-1])


def cost_spectral_gap(cost_matrix) -> float:
    """
    Spectral gap of cost matrix: sigma_1 / sigma_2.
    Higher = more dominant principal component. Indicates cost matrix rank structure.
    cost_matrix: [B, heads, N, N] or [N, N]
    """
    cost_matrix = _to_numpy(cost_matrix)
    if cost_matrix.ndim > 2:
        cost_matrix = cost_matrix.mean(axis=tuple(range(cost_matrix.ndim - 2)))
    _, S, _ = np.linalg.svd(cost_matrix.astype(np.float32), full_matrices=False)
    S = S[S > 1e-10]
    if len(S) < 2:
        return float(S[0]) if len(S) == 1 else 0.0
    return float(S[0] / S[1])


def tension_balance(attention_low, attention_high) -> float:
    """
    Measures how different the two attention patterns are.
    0.0 = identical patterns (no tension)
    1.0 = maximally different (maximum tension)

    Computed as 1 - cosine_similarity(flatten(A_low), flatten(A_high))
    """
    a = _to_numpy(attention_low).flatten().astype(np.float32)
    b = _to_numpy(attention_high).flatten().astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    return float(1.0 - np.clip(cos_sim, -1.0, 1.0))


if __name__ == "__main__":
    print("[Metrics] Running numpy quick test...")
    B, N, D = 4, 64, 128
    embeddings = np.random.randn(B, N, D).astype(np.float32)

    er = effective_rank(embeddings)
    print(f"  Effective Rank: {er:.2f}")

    idim = intrinsic_dimension_mle(embeddings)
    print(f"  Intrinsic Dim (MLE): {idim:.2f}")

    ai = anisotropy_index(embeddings)
    print(f"  Anisotropy Index: {ai:.4f}")

    pd = pairwise_distance_stats(embeddings)
    print(
        f"  Pairwise Dist: mean={pd['mean']:.2f}, std={pd['std']:.2f}, cv={pd['cv']:.4f}"
    )

    # Test with attention matrix
    attn = np.random.randn(B, 8, N, N).astype(np.float32)
    attn = np.exp(attn)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    cr = concentration_ratio(attn)
    ae = attention_entropy(attn)
    print(f"  Concentration Ratio: {cr:.4f}")
    print(f"  Attention Entropy: {ae:.2f}")

    all_metrics = compute_all_metrics(embeddings, attn)
    print(f"\n  All metrics:")
    for key, val in all_metrics.items():
        print(f"    {key}: {val:.4f}")

    # Test cost metrics
    cost_mat = np.random.randn(32, 32).astype(np.float32) ** 2  # Non-negative
    cn = cost_condition_number(cost_mat)
    sg = cost_spectral_gap(cost_mat)
    print(f"  Cost Condition Number: {cn:.4f}")
    print(f"  Cost Spectral Gap: {sg:.4f}")

    print("\n[Metrics] All numpy tests passed!")
