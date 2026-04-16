"""
Spectral Metrics — Gram Matrix Analysis for Collapse Detection
===============================================================
Based on Sigma paper (arXiv:2601.03385): Scalable Spectral Insights for LLM Model Collapse

Metrics for detecting representation collapse via eigenspectrum of Gram matrix.

Pure NumPy implementation.
"""

import numpy as np
from typing import Dict


def compute_gram_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute Gram matrix G = X @ X.T"""
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    embeddings = embeddings.astype(np.float32)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    G = centered @ centered.T
    return G


def spectral_log_det(gram_matrix: np.ndarray) -> float:
    """Log-determinant of Gram matrix: primary metric for model collapse"""
    eigenvalues = np.linalg.eigvalsh(gram_matrix.astype(np.float32))
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(positive_eigenvalues) == 0:
        return -np.inf
    log_det = np.sum(np.log(positive_eigenvalues))
    return float(log_det)


def spectral_bounds(gram_matrix: np.ndarray, n_samples: int = 100) -> Dict[str, float]:
    """Stochastic spectral bounds for scalable collapse detection"""
    N = gram_matrix.shape[0]
    rng = np.random.RandomState(42)
    trace_sum = 0.0
    for _ in range(n_samples):
        v = rng.randn(N)
        v = v / np.linalg.norm(v)
        trace_sum += v.T @ gram_matrix @ v
    trace_estimate = trace_sum / n_samples
    trace_deterministic = np.trace(gram_matrix)
    gershgorin_lower = np.max(np.diag(gram_matrix))
    return {
        "trace_estimate": float(trace_estimate),
        "trace_deterministic": float(trace_deterministic),
        "gershgorin_lower": float(gershgorin_lower),
    }


def iso_score(embeddings: np.ndarray, k: int = 10) -> float:
    """Isotropy score: fraction of variance explained by top-k eigenvalues"""
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / centered.shape[0]
    eigenvalues = np.linalg.eigvalsh(cov.astype(np.float32))
    eigenvalues = np.sort(eigenvalues)[::-1]
    total_variance = np.sum(eigenvalues)
    if total_variance < 1e-10:
        return 0.0
    top_k_variance = np.sum(eigenvalues[:k])
    return float(top_k_variance / total_variance)


def spectral_decay_rate(embeddings: np.ndarray) -> float:
    """Estimate spectral decay rate alpha in: lambda_i ~ i^(-alpha)"""
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    S = S[S > 1e-10]
    if len(S) < 3:
        return 1.0
    log_i = np.log(np.arange(1, len(S) + 1))
    log_S = np.log(S + 1e-10)
    alpha = -np.polyfit(log_i, log_S, 1)[0]
    return float(max(0.0, min(5.0, alpha)))


def collapse_score(gram_matrix: np.ndarray) -> float:
    """Composite collapse score combining multiple spectral metrics"""
    log_det = spectral_log_det(gram_matrix)
    N = gram_matrix.shape[0]
    expected_log_det = N * np.log(N * 0.1)
    log_det_normalized = np.clip(log_det / expected_log_det, -2.0, 1.0)
    log_det_score = 1.0 - np.clip((log_det_normalized + 1.0) / 2.0, 0.0, 1.0)

    eigenvalues = np.linalg.eigvalsh(gram_matrix.astype(np.float32))
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(positive_eigenvalues) >= 2:
        cond = positive_eigenvalues[-1] / positive_eigenvalues[0]
        cond_score = 1.0 - 1.0 / (1.0 + cond)
    else:
        cond_score = 0.5

    if len(positive_eigenvalues) >= 2:
        spread = positive_eigenvalues.max() / (positive_eigenvalues.min() + 1e-10)
        spread_score = 1.0 - 1.0 / (1.0 + spread)
    else:
        spread_score = 0.5

    collapse = 0.4 * log_det_score + 0.3 * cond_score + 0.3 * spread_score
    return float(np.clip(collapse, 0.0, 1.0))


def compute_all_spectral_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute all spectral metrics in one call"""
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    G = compute_gram_matrix(embeddings)
    log_det = spectral_log_det(G)
    bounds = spectral_bounds(G)
    iso = iso_score(embeddings)
    decay = spectral_decay_rate(embeddings)
    collapse = collapse_score(G)
    return {
        "spectral_log_det": log_det,
        "trace_estimate": bounds["trace_estimate"],
        "trace_deterministic": bounds["trace_deterministic"],
        "iso_score_k10": iso,
        "spectral_decay_rate": decay,
        "collapse_score": collapse,
    }


if __name__ == "__main__":
    np.random.seed(42)
    N, D = 100, 64
    random_emb = np.random.randn(N, D).astype(np.float32)
    random_emb = random_emb / np.linalg.norm(random_emb, axis=1, keepdims=True)
    metrics = compute_all_spectral_metrics(random_emb)
    print("=== Spectral Metrics Test ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
