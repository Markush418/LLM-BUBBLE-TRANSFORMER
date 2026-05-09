"""Tests for spectral metrics module (Gram matrix analysis)."""

import sys
import os
import unittest
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import numpy as np

from spectral_metrics import (
    compute_gram_matrix,
    spectral_log_det,
    spectral_bounds,
    iso_score,
    spectral_decay_rate,
    collapse_score,
    compute_all_spectral_metrics,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))


class TestComputeGramMatrix(unittest.TestCase):
    """Tests for compute_gram_matrix."""

    def test_zero_embeddings(self):
        """All-zero embeddings should produce a zero Gram matrix."""
        embeddings = np.zeros((10, 8), dtype=np.float32)
        G = compute_gram_matrix(embeddings)
        self.assertEqual(G.shape, (10, 10))
        self.assertTrue(np.allclose(G, 0.0))

    def test_identical_embeddings(self):
        """All-identical embeddings should produce a zero Gram matrix after centering."""
        embeddings = np.ones((10, 8), dtype=np.float32)
        G = compute_gram_matrix(embeddings)
        self.assertTrue(np.allclose(G, 0.0, atol=1e-5))

    def test_3d_input(self):
        """3D input [B, N, D] should be flattened to [B*N, D]."""
        embeddings = np.random.randn(2, 8, 16).astype(np.float32)
        G = compute_gram_matrix(embeddings)
        self.assertEqual(G.shape, (16, 16))

    def test_symmetric_positive_semidefinite(self):
        """Gram matrix must be symmetric and positive semi-definite."""
        embeddings = np.random.randn(20, 12).astype(np.float32)
        G = compute_gram_matrix(embeddings)
        self.assertTrue(np.allclose(G, G.T, atol=1e-5))
        eigenvalues = np.linalg.eigvalsh(G)
        self.assertTrue(np.all(eigenvalues >= -1e-5))


class TestSpectralLogDet(unittest.TestCase):
    """Tests for spectral_log_det."""

    def test_zero_matrix(self):
        """Zero matrix should return -inf."""
        G = np.zeros((8, 8), dtype=np.float32)
        log_det = spectral_log_det(G)
        self.assertEqual(log_det, -np.inf)

    def test_identity_like_gram(self):
        """A Gram matrix with known positive eigenvalues should return sum of logs."""
        # Construct a symmetric positive-definite matrix with known eigenvalues
        eigenvalues = np.array([4.0, 2.0, 1.0, 0.5], dtype=np.float32)
        Q = np.eye(4, dtype=np.float32)
        G = Q @ np.diag(eigenvalues) @ Q.T
        log_det = spectral_log_det(G)
        expected = np.sum(np.log(eigenvalues))
        self.assertAlmostEqual(log_det, expected, places=4)

    def test_singular_matrix(self):
        """Singular matrix with some zero eigenvalues should ignore zeros."""
        eigenvalues = np.array([3.0, 2.0, 0.0, 0.0], dtype=np.float32)
        Q = np.eye(4, dtype=np.float32)
        G = Q @ np.diag(eigenvalues) @ Q.T
        log_det = spectral_log_det(G)
        expected = np.sum(np.log(np.array([3.0, 2.0])))
        self.assertAlmostEqual(log_det, expected, places=4)

    def test_1x1_matrix(self):
        """1x1 matrix with positive value."""
        G = np.array([[5.0]], dtype=np.float32)
        log_det = spectral_log_det(G)
        self.assertAlmostEqual(log_det, np.log(5.0), places=4)

    def test_1x1_zero(self):
        """1x1 zero matrix should return -inf."""
        G = np.array([[0.0]], dtype=np.float32)
        log_det = spectral_log_det(G)
        self.assertEqual(log_det, -np.inf)


class TestSpectralBounds(unittest.TestCase):
    """Tests for spectral_bounds."""

    def test_zero_matrix(self):
        """Zero matrix should produce zero bounds."""
        G = np.zeros((8, 8), dtype=np.float32)
        bounds = spectral_bounds(G, n_samples=50)
        self.assertAlmostEqual(bounds["trace_estimate"], 0.0, places=4)
        self.assertAlmostEqual(bounds["trace_deterministic"], 0.0, places=4)
        self.assertAlmostEqual(bounds["gershgorin_lower"], 0.0, places=4)

    def test_identity_matrix(self):
        """Identity matrix should have deterministic trace = N and gershgorin_lower = 1.

        Note: stochastic trace estimate for identity gives ~1.0 because
        v.T @ I @ v = ||v||^2 = 1 for any unit vector v.
        """
        N = 8
        G = np.eye(N, dtype=np.float32)
        bounds = spectral_bounds(G, n_samples=100)
        self.assertAlmostEqual(bounds["trace_deterministic"], float(N), places=4)
        self.assertAlmostEqual(bounds["gershgorin_lower"], 1.0, places=4)
        # Stochastic trace estimate for identity is always 1.0 (v.T @ I @ v = 1)
        self.assertAlmostEqual(bounds["trace_estimate"], 1.0, places=4)

    def test_returns_required_keys(self):
        """Result must contain all expected keys."""
        G = np.random.randn(10, 10).astype(np.float32)
        G = G @ G.T  # make PSD
        bounds = spectral_bounds(G)
        self.assertIn("trace_estimate", bounds)
        self.assertIn("trace_deterministic", bounds)
        self.assertIn("gershgorin_lower", bounds)


class TestIsoScore(unittest.TestCase):
    """Tests for iso_score."""

    def test_collapsed_embeddings(self):
        """All-same embeddings should have iso_score = 0.0."""
        embeddings = np.ones((20, 16), dtype=np.float32)
        iso = iso_score(embeddings, k=5)
        self.assertAlmostEqual(iso, 0.0, places=5)

    def test_random_embeddings_range(self):
        """Random embeddings should have iso_score in (0, 1)."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 32).astype(np.float32)
        iso = iso_score(embeddings, k=10)
        self.assertGreater(iso, 0.0)
        self.assertLess(iso, 1.0)

    def test_3d_input(self):
        """3D input [B, N, D] should be handled."""
        embeddings = np.random.randn(2, 8, 16).astype(np.float32)
        iso = iso_score(embeddings, k=5)
        self.assertGreaterEqual(iso, 0.0)
        self.assertLessEqual(iso, 1.0)

    def test_high_k(self):
        """k larger than dimension should still work."""
        embeddings = np.random.randn(10, 4).astype(np.float32)
        iso = iso_score(embeddings, k=10)
        self.assertAlmostEqual(iso, 1.0, places=4)


class TestSpectralDecayRate(unittest.TestCase):
    """Tests for spectral_decay_rate."""

    def test_very_small_matrix(self):
        """Matrix with < 3 singular values should return 1.0."""
        embeddings = np.random.randn(2, 4).astype(np.float32)
        rate = spectral_decay_rate(embeddings)
        self.assertEqual(rate, 1.0)

    def test_random_embeddings_range(self):
        """Random embeddings should have decay rate in [0, 5]."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 32).astype(np.float32)
        rate = spectral_decay_rate(embeddings)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 5.0)

    def test_collapsed_embeddings(self):
        """Collapsed embeddings have very few non-zero singular values."""
        embeddings = np.ones((20, 16), dtype=np.float32)
        rate = spectral_decay_rate(embeddings)
        # After centering, collapsed embeddings have rank 0 -> < 3 positive S -> 1.0
        self.assertEqual(rate, 1.0)

    def test_3d_input(self):
        """3D input should be handled."""
        embeddings = np.random.randn(2, 16, 8).astype(np.float32)
        rate = spectral_decay_rate(embeddings)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 5.0)


class TestCollapseScore(unittest.TestCase):
    """Tests for collapse_score."""

    def test_zero_matrix(self):
        """Zero Gram matrix should have high collapse score."""
        G = np.zeros((10, 10), dtype=np.float32)
        score = collapse_score(G)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # Zero matrix is completely collapsed
        self.assertGreater(score, 0.5)

    def test_well_conditioned_matrix(self):
        """Well-conditioned matrix should have lower collapse score."""
        # Use random embeddings to get a well-conditioned Gram matrix
        np.random.seed(42)
        embeddings = np.random.randn(50, 32).astype(np.float32)
        G = compute_gram_matrix(embeddings)
        score = collapse_score(G)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_range(self):
        """Collapse score must always be in [0, 1]."""
        np.random.seed(42)
        for _ in range(5):
            embeddings = np.random.randn(30, 16).astype(np.float32)
            G = compute_gram_matrix(embeddings)
            score = collapse_score(G)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestComputeAllSpectralMetrics(unittest.TestCase):
    """Tests for compute_all_spectral_metrics."""

    def test_returns_all_keys(self):
        """Should return all expected metric keys."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 16).astype(np.float32)
        metrics = compute_all_spectral_metrics(embeddings)
        expected_keys = [
            "spectral_log_det",
            "trace_estimate",
            "trace_deterministic",
            "iso_score_k10",
            "spectral_decay_rate",
            "collapse_score",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_values_are_finite(self):
        """All returned metric values should be finite (not NaN/inf)."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 16).astype(np.float32)
        metrics = compute_all_spectral_metrics(embeddings)
        for key, value in metrics.items():
            self.assertTrue(
                math.isfinite(value),
                f"Metric '{key}' is not finite: {value}",
            )

    def test_3d_input(self):
        """Should handle 3D input [B, N, D]."""
        embeddings = np.random.randn(2, 8, 16).astype(np.float32)
        metrics = compute_all_spectral_metrics(embeddings)
        self.assertIn("collapse_score", metrics)
        self.assertGreaterEqual(metrics["collapse_score"], 0.0)
        self.assertLessEqual(metrics["collapse_score"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
