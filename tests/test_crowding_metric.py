"""Tests for embedding space crowding metrics module.

All fixtures use NumPy arrays (no PyTorch) to match the production-code
contract of ``crowding_metric.py``.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import numpy as np
from crowding_metric import (
    compute_pairwise_distances,
    crowding_ratio,
    mean_nearest_neighbor_distance,
    clustering_coefficient,
    space_coverage,
    anisotropy_index,
    compute_all_crowding_metrics,
)
from tests.test_helpers import assert_numpy_close


class TestComputePairwiseDistances(unittest.TestCase):
    """Tests for pairwise distance computation."""

    def test_identity_distance(self):
        """Identical embeddings should have zero distance."""
        embeddings = np.ones((10, 8), dtype=np.float32)
        distances = compute_pairwise_distances(embeddings)
        np.fill_diagonal(distances, 0.0)
        self.assertTrue(np.allclose(distances, 0.0, atol=1e-6))

    def test_3d_input_flattening(self):
        """Should flatten [B, N, D] to [B*N, D]."""
        emb_3d = np.random.randn(2, 8, 16).astype(np.float32)
        distances_3d = compute_pairwise_distances(emb_3d)
        emb_2d = emb_3d.reshape(-1, emb_3d.shape[-1])
        distances_2d = compute_pairwise_distances(emb_2d)
        assert_numpy_close(distances_3d, distances_2d)

    def test_distance_range(self):
        """Cosine distances should be in [0, 2]."""
        embeddings = np.random.randn(20, 16).astype(np.float32)
        distances = compute_pairwise_distances(embeddings)
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.all(distances <= 2))

    def test_symmetry(self):
        """Distance matrix should be symmetric."""
        embeddings = np.random.randn(15, 12).astype(np.float32)
        distances = compute_pairwise_distances(embeddings)
        assert_numpy_close(distances, distances.T)


class TestCrowdingRatio(unittest.TestCase):
    """Tests for crowding ratio metric."""

    def test_clustered_vs_uniform(self):
        """Clustered embeddings should have higher crowding ratio."""
        np.random.seed(42)
        # Uniform: well spread on unit sphere
        uniform = np.random.randn(50, 32).astype(np.float32)
        uniform /= np.linalg.norm(uniform, axis=1, keepdims=True)

        # Clustered: two tight clusters
        cluster1 = np.random.randn(25, 32).astype(np.float32) * 0.01 + 1.0
        cluster2 = np.random.randn(25, 32).astype(np.float32) * 0.01 - 1.0
        clustered = np.vstack([cluster1, cluster2])

        cr_uniform = crowding_ratio(uniform, k=5)
        cr_clustered = crowding_ratio(clustered, k=5)
        self.assertGreater(
            cr_clustered,
            cr_uniform,
            f"Clustered CR ({cr_clustered}) should exceed uniform CR ({cr_uniform})",
        )

    def test_single_point(self):
        """Single point should have crowding ratio 0.0."""
        embeddings = np.random.randn(1, 16).astype(np.float32)
        cr = crowding_ratio(embeddings, k=1)
        self.assertEqual(cr, 0.0)

    def test_all_identical(self):
        """All identical points should have maximum crowding."""
        embeddings = np.ones((20, 8), dtype=np.float32)
        cr = crowding_ratio(embeddings, k=10)
        self.assertEqual(cr, 1.0)

    def test_empty_input(self):
        """Empty input should raise ZeroDivisionError (documented edge case)."""
        embeddings = np.zeros((0, 16), dtype=np.float32)
        with self.assertRaises(ZeroDivisionError):
            crowding_ratio(embeddings, k=1)


class TestMeanNearestNeighborDistance(unittest.TestCase):
    """Tests for mean nearest-neighbor distance."""

    def test_identical_points_zero_distance(self):
        """Identical points should have zero mean NN distance."""
        embeddings = np.ones((10, 8), dtype=np.float32)
        dist = mean_nearest_neighbor_distance(embeddings)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_clustered_vs_uniform(self):
        """Clustered embeddings should have smaller mean NN distance."""
        np.random.seed(42)
        uniform = np.random.randn(50, 32).astype(np.float32)
        uniform /= np.linalg.norm(uniform, axis=1, keepdims=True)

        cluster1 = np.random.randn(25, 32).astype(np.float32) * 0.01 + 1.0
        cluster2 = np.random.randn(25, 32).astype(np.float32) * 0.01 - 1.0
        clustered = np.vstack([cluster1, cluster2])

        dist_uniform = mean_nearest_neighbor_distance(uniform)
        dist_clustered = mean_nearest_neighbor_distance(clustered)
        self.assertLess(
            dist_clustered,
            dist_uniform,
            f"Clustered NN dist ({dist_clustered}) should be < uniform ({dist_uniform})",
        )

    def test_two_points(self):
        """Two points should return their cosine distance."""
        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        embeddings = np.vstack([a, b])
        dist = mean_nearest_neighbor_distance(embeddings)
        # Cosine distance between orthogonal vectors = 1.0
        self.assertAlmostEqual(dist, 1.0, places=5)


class TestClusteringCoefficient(unittest.TestCase):
    """Tests for local clustering coefficient."""

    def test_high_clustering(self):
        """Tight clusters should have high clustering coefficient."""
        np.random.seed(42)
        cluster1 = np.random.randn(15, 16).astype(np.float32) * 0.01 + 1.0
        cluster2 = np.random.randn(15, 16).astype(np.float32) * 0.01 - 1.0
        clustered = np.vstack([cluster1, cluster2])
        coef = clustering_coefficient(clustered, k=5)
        self.assertGreater(coef, 0.0, f"Expected positive clustering, got {coef}")

    def test_uniform_low_clustering(self):
        """Uniform random should have low clustering coefficient."""
        np.random.seed(42)
        uniform = np.random.randn(100, 32).astype(np.float32)
        uniform /= np.linalg.norm(uniform, axis=1, keepdims=True)
        coef = clustering_coefficient(uniform, k=5)
        # Uniform random on sphere has low local clustering
        self.assertLess(coef, 0.5, f"Expected low clustering for uniform, got {coef}")

    def test_single_point(self):
        """Single point should return 0.0 (no neighbors)."""
        embeddings = np.random.randn(1, 16).astype(np.float32)
        coef = clustering_coefficient(embeddings, k=1)
        self.assertEqual(coef, 0.0)

    def test_duplicate_points(self):
        """Duplicate points should have high local clustering."""
        embeddings = np.ones((10, 8), dtype=np.float32)
        coef = clustering_coefficient(embeddings, k=3)
        self.assertEqual(coef, 1.0)


class TestSpaceCoverage(unittest.TestCase):
    """Tests for space coverage metric."""

    def test_well_spread_coverage(self):
        """Well-spread embeddings should cover more bins."""
        np.random.seed(42)
        spread = np.random.randn(200, 32).astype(np.float32)
        spread /= np.linalg.norm(spread, axis=1, keepdims=True)
        cov = space_coverage(spread, n_bins=10)
        self.assertGreater(cov, 0.0)
        self.assertLessEqual(cov, 1.0)

    def test_clustered_low_coverage(self):
        """Clustered embeddings should have lower coverage than spread."""
        np.random.seed(42)
        spread = np.random.randn(200, 32).astype(np.float32)
        spread /= np.linalg.norm(spread, axis=1, keepdims=True)

        cluster = np.random.randn(200, 32).astype(np.float32) * 0.01
        cluster += np.array([1.0, -1.0, 0.0] + [0.0] * 29, dtype=np.float32)

        cov_spread = space_coverage(spread, n_bins=10)
        cov_cluster = space_coverage(cluster, n_bins=10)
        self.assertGreater(
            cov_spread,
            cov_cluster,
            f"Spread coverage ({cov_spread}) should exceed clustered ({cov_cluster})",
        )

    def test_3d_input(self):
        """Should handle [B, N, D] input."""
        emb_3d = np.random.randn(2, 16, 8).astype(np.float32)
        cov = space_coverage(emb_3d, n_bins=5)
        self.assertGreaterEqual(cov, 0.0)
        self.assertLessEqual(cov, 1.0)


class TestAnisotropyIndex(unittest.TestCase):
    """Tests for anisotropy index."""

    def test_isotropic(self):
        """Random Gaussian should have low anisotropy."""
        np.random.seed(42)
        embeddings = np.random.randn(500, 32).astype(np.float32)
        ai = anisotropy_index(embeddings)
        self.assertLess(ai, 0.5, f"Expected low anisotropy, got {ai}")

    def test_anisotropic(self):
        """Embeddings stretched along one axis should have high anisotropy."""
        np.random.seed(42)
        embeddings = np.random.randn(200, 32).astype(np.float32)
        embeddings[:, 0] *= 100.0
        ai = anisotropy_index(embeddings)
        self.assertGreater(ai, 0.5, f"Expected high anisotropy, got {ai}")

    def test_3d_input(self):
        """Should handle [B, N, D] input."""
        emb_3d = np.random.randn(2, 16, 8).astype(np.float32)
        ai = anisotropy_index(emb_3d)
        self.assertGreater(ai, 0.0)
        self.assertLessEqual(ai, 1.0)

    def test_all_zeros(self):
        """All-zero embeddings should return 1.0 (max anisotropy fallback)."""
        embeddings = np.zeros((10, 8), dtype=np.float32)
        ai = anisotropy_index(embeddings)
        self.assertEqual(ai, 1.0)


class TestComputeAllCrowdingMetrics(unittest.TestCase):
    """Tests for the combined metrics function."""

    def test_returns_all_keys(self):
        """Should return all expected metric keys."""
        embeddings = np.random.randn(50, 32).astype(np.float32)
        metrics = compute_all_crowding_metrics(embeddings)
        expected_keys = [
            "crowding_ratio_k10",
            "mean_nearest_neighbor_dist",
            "clustering_coefficient",
            "space_coverage",
            "anisotropy_index",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_values_are_finite(self):
        """All returned values should be finite numbers."""
        embeddings = np.random.randn(50, 32).astype(np.float32)
        metrics = compute_all_crowding_metrics(embeddings)
        for key, value in metrics.items():
            self.assertTrue(
                np.isfinite(value),
                f"Metric '{key}' is non-finite: {value}",
            )

    def test_empty_input(self):
        """Empty input should raise ZeroDivisionError (documented edge case)."""
        embeddings = np.zeros((0, 16), dtype=np.float32)
        with self.assertRaises(ZeroDivisionError):
            compute_all_crowding_metrics(embeddings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
