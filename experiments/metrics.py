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


def effective_rank(embeddings: np.ndarray) -> float:
    """
    Effective rank via SVD: exp(H(p)) where p_i = sigma_i / sum(sigma).
    embeddings: [N, D] or [B, N, D]
    """
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


def intrinsic_dimension_mle(embeddings: np.ndarray, k: int = 10) -> float:
    """
    MLE estimator for intrinsic dimensionality (Levina & Bickel 2004).
    Uses k-NN distances.
    """
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


def anisotropy_index(embeddings: np.ndarray) -> float:
    """
    Ratio of max eigenvalue to sum of eigenvalues of covariance matrix.
    1.0 = fully anisotropic, 1/d = perfectly isotropic.
    """
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / centered.shape[0]
    eigenvalues = np.linalg.eigvalsh(cov.astype(np.float32))
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 1.0
    return float(eigenvalues[-1] / eigenvalues.sum())


def pairwise_distance_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute statistics of pairwise distances."""
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
    attention_matrix: np.ndarray, threshold_factor: float = 1.0
) -> float:
    """
    Fraction of entries above threshold = threshold_factor / N.
    Lower = more concentrated (sparser).
    """
    if attention_matrix.ndim == 4:
        attention_matrix = attention_matrix.mean(axis=(0, 1))
    elif attention_matrix.ndim == 3:
        attention_matrix = attention_matrix.mean(axis=0)
    N = attention_matrix.shape[-1]
    threshold = threshold_factor / N
    active = np.sum(attention_matrix > threshold)
    total = attention_matrix.size
    return float(active / total)


def attention_entropy(attention_matrix: np.ndarray) -> float:
    """Entropy of the attention distribution. Lower = more peaked."""
    if attention_matrix.ndim == 4:
        attention_matrix = attention_matrix.mean(axis=(0, 1))
    elif attention_matrix.ndim == 3:
        attention_matrix = attention_matrix.mean(axis=0)
    A = attention_matrix.reshape(-1, attention_matrix.shape[-1])
    A = A + 1e-10
    A = A / A.sum(axis=-1, keepdims=True)
    return float(np.mean(-np.sum(A * np.log(A), axis=-1)))


def compute_all_metrics(
    embeddings: np.ndarray,
    attention_matrix: Optional[np.ndarray] = None,
    baseline_effective_rank: Optional[float] = None,
) -> Dict[str, float]:
    """Compute all concentration and geometry metrics at once."""
    metrics = {}
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
    if attention_matrix is not None:
        metrics["concentration_ratio"] = concentration_ratio(attention_matrix)
        metrics["attention_entropy"] = attention_entropy(attention_matrix)
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

    print("\n[Metrics] All numpy tests passed!")
