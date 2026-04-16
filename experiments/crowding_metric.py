"""
Crowding Metrics — Embedding Space Crowding Analysis
================================================

Based on research on embedding-space crowding (arXiv:2305.xxxxx):
- Crowding: when multiple tokens map to similar embedding regions
- Affects decoding quality and attention diversity
- Measure using nearest-neighbor statistics

Pure NumPy implementation.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.spatial.distance import cdist


def compute_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distances between embeddings.

    Handles both [N, D] and [B, N, D] shapes.
    """
    # Handle 3D input by flattening batch
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Cosine distance = 1 - cosine_similarity
    distances = 1.0 - np.dot(normalized, normalized.T)
    return np.clip(distances, 0, 2)


def crowding_ratio(embeddings: np.ndarray, k: int = 10) -> float:
    """
    Crowding ratio: fraction of tokens with >= k nearest neighbors within threshold.

    High crowding = many tokens packed in similar regions.
    Low crowding = well-distributed embeddings.
    """
    distances = compute_pairwise_distances(embeddings)

    # For each embedding, count neighbors within threshold
    threshold = 0.1  # Cosine distance threshold
    n_samples = embeddings.shape[0]

    crowded_count = 0
    for i in range(n_samples):
        neighbors_within_threshold = (
            np.sum(distances[i] < threshold) - 1
        )  # Exclude self
        if neighbors_within_threshold >= k:
            crowded_count += 1

    return crowded_count / n_samples


def mean_nearest_neighbor_distance(embeddings: np.ndarray) -> float:
    """Mean distance to nearest neighbor for each embedding"""
    distances = compute_pairwise_distances(embeddings)

    # Set diagonal to infinity to exclude self
    np.fill_diagonal(distances, np.inf)

    # Mean of minimum distances
    mean_min_dist = np.mean(np.min(distances, axis=1))
    return float(mean_min_dist)


def clustering_coefficient(embeddings: np.ndarray, k: int = 10) -> float:
    """
    Local clustering coefficient based on k-nearest neighbors.

    High clustering = embeddings form tight local groups.
    """
    distances = compute_pairwise_distances(embeddings)
    n = embeddings.shape[0]

    total_coef = 0.0
    for i in range(n):
        # Get k nearest neighbors
        neighbor_indices = np.argsort(distances[i])[: k + 1][1:]  # Exclude self

        # Count edges between neighbors
        edges = 0
        for j in neighbor_indices:
            for l in neighbor_indices:
                if j < l and distances[j, l] < 0.1:  # Threshold for "connected"
                    edges += 1

        # Local clustering coefficient
        max_edges = k * (k - 1) / 2
        if max_edges > 0:
            local_coef = edges / max_edges
            total_coef += local_coef

    return total_coef / n


def space_coverage(embeddings: np.ndarray, n_bins: int = 20) -> float:
    """
    Fraction of embedding space volume covered by embeddings.

    Uses angular quantization to estimate coverage.
    """
    # Handle 3D input by flattening batch
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    # Normalize to unit sphere
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Project to 2D using first two principal components
    centered = normalized - normalized.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    top_2_eigenvectors = eigenvectors[:, -2:]
    projected_2d = centered @ top_2_eigenvectors

    # Bin the 2D projections
    x_min, x_max = projected_2d[:, 0].min(), projected_2d[:, 0].max()
    y_min, y_max = projected_2d[:, 1].min(), projected_2d[:, 1].max()

    x_bins = np.linspace(x_min, x_max, n_bins)
    y_bins = np.linspace(y_min, y_max, n_bins)

    # Count occupied bins
    occupied = set()
    for x, y in projected_2d:
        x_idx = np.searchsorted(x_bins, x)
        y_idx = np.searchsorted(y_bins, y)
        x_idx = min(x_idx, n_bins - 1)
        y_idx = min(y_idx, n_bins - 1)
        occupied.add((x_idx, y_idx))

    total_bins = n_bins * n_bins
    return len(occupied) / total_bins


def anisotropy_index(embeddings: np.ndarray) -> float:
    """
    Anisotropy index: ratio of max eigenvalue to sum of eigenvalues.

    1.0 = maximum anisotropy (collapse to single direction)
    ~1/n = isotropic (equal variance in all directions)
    """
    # Handle 3D input by flattening batch
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]

    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0

    max_eigenvalue = eigenvalues[0]
    sum_eigenvalues = np.sum(eigenvalues)

    return float(max_eigenvalue / sum_eigenvalues)


def compute_all_crowding_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute all crowding metrics in one call"""
    return {
        "crowding_ratio_k10": crowding_ratio(embeddings, k=10),
        "mean_nearest_neighbor_dist": mean_nearest_neighbor_distance(embeddings),
        "clustering_coefficient": clustering_coefficient(embeddings, k=10),
        "space_coverage": space_coverage(embeddings),
        "anisotropy_index": anisotropy_index(embeddings),
    }


if __name__ == "__main__":
    np.random.seed(42)
    N, D = 100, 64

    # Test with random embeddings
    random_emb = np.random.randn(N, D).astype(np.float32)
    random_emb = random_emb / np.linalg.norm(random_emb, axis=1, keepdims=True)

    metrics = compute_all_crowding_metrics(random_emb)
    print("=== Crowding Metrics Test ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
