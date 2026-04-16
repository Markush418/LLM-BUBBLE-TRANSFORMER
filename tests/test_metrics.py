"""Tests for concentration/geometry metrics module."""

import sys
import os
import math
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from metrics import (
    effective_rank,
    intrinsic_dimension_mle,
    anisotropy_index,
    pairwise_distance_stats,
    concentration_ratio,
    attention_entropy,
    compute_all_metrics,
)


class TestEffectiveRank(unittest.TestCase):
    """Tests for the effective rank metric."""

    def test_full_rank_matrix(self):
        """Identity-like matrix should have high effective rank."""
        embeddings = torch.eye(64) * 10
        rank = effective_rank(embeddings)
        self.assertGreater(rank, 30, f"Expected high rank, got {rank}")

    def test_collapsed_embeddings(self):
        """All-same embeddings should have very low effective rank."""
        embeddings = torch.ones(64, 128)
        rank = effective_rank(embeddings)
        self.assertLess(rank, 5, f"Expected near-zero rank, got {rank}")

    def test_random_embeddings(self):
        """Random embeddings should have moderate effective rank."""
        embeddings = torch.randn(100, 128)
        rank = effective_rank(embeddings)
        self.assertGreater(rank, 10, f"Expected rank > 10, got {rank}")
        self.assertLess(rank, 150, f"Expected rank < 150, got {rank}")

    def test_3d_input(self):
        """Should handle [B, N, D] input by flattening."""
        embeddings = torch.randn(4, 32, 128)
        rank = effective_rank(embeddings)
        self.assertGreater(rank, 0)


class TestIntrinsicDimensionMLE(unittest.TestCase):
    """Tests for the MLE intrinsic dimensionality estimator."""

    def test_low_dim_manifold(self):
        """Points on a line should have intrinsic dim ~1."""
        t = torch.linspace(0, 10, 100).unsqueeze(1)
        embeddings = torch.cat([t, torch.zeros(100, 127)], dim=1)
        embeddings += torch.randn_like(embeddings) * 1e-6
        dim = intrinsic_dimension_mle(embeddings, k=5)
        self.assertLess(dim, 20, f"Expected low dim, got {dim}")

    def test_high_dim_random(self):
        """Random Gaussian should have high intrinsic dimensionality."""
        embeddings = torch.randn(200, 128)
        dim = intrinsic_dimension_mle(embeddings, k=10)
        self.assertGreater(dim, 20, f"Expected high dim, got {dim}")


class TestAnisotropyIndex(unittest.TestCase):
    """Tests for the anisotropy index."""

    def test_isotropic_embeddings(self):
        """Random Gaussian should have low anisotropy."""
        embeddings = torch.randn(500, 64)
        ai = anisotropy_index(embeddings)
        self.assertLess(ai, 0.1, f"Expected low anisotropy, got {ai}")

    def test_anisotropic_embeddings(self):
        """Embeddings along one direction should have high anisotropy."""
        embeddings = torch.randn(100, 64)
        embeddings[:, 0] *= 100
        ai = anisotropy_index(embeddings)
        self.assertGreater(ai, 0.5, f"Expected high anisotropy, got {ai}")


class TestPairwiseDistanceStats(unittest.TestCase):
    """Tests for pairwise distance statistics."""

    def test_identical_points(self):
        """All identical points should have zero distances."""
        embeddings = torch.ones(50, 32)
        stats = pairwise_distance_stats(embeddings)
        self.assertAlmostEqual(stats["mean"], 0.0, places=5)

    def test_spread_points(self):
        """Well-spread points should have positive mean distance."""
        embeddings = torch.randn(100, 32) * 10
        stats = pairwise_distance_stats(embeddings)
        self.assertGreater(stats["mean"])
        self.assertGreater(stats["std"])


class TestConcentrationRatio(unittest.TestCase):
    """Tests for the concentration ratio metric."""

    def test_uniform_attention(self):
        """Uniform attention should have concentration ratio = 1.0."""
        N = 32
        attn = torch.ones(1, 1, N, N) / N
        cr = concentration_ratio(attn)
        self.assertAlmostEqual(cr, 1.0, places=2)

    def test_one_hot_attention(self):
        """One-hot attention should have very low concentration ratio."""
        N = 32
        attn = torch.zeros(1, 1, N, N)
        attn[:, :, :, 0] = 1.0
        cr = concentration_ratio(attn)
        self.assertLess(cr, 0.1, f"Expected low CR for one-hot, got {cr}")


class TestAttentionEntropy(unittest.TestCase):
    """Tests for attention entropy metric."""

    def test_uniform_entropy(self):
        """Uniform distribution should have maximum entropy."""
        N = 32
        attn = torch.ones(1, 1, N, N) / N
        ent = attention_entropy(attn)
        expected = math.log(N)
        self.assertAlmostEqual(ent, expected, places=1)

    def test_peaked_entropy(self):
        """Peaked distribution should have low entropy."""
        N = 32
        attn = torch.zeros(1, 1, N, N)
        attn[:, :, 0, 0] = 1.0
        attn = attn + 1e-10
        attn = attn / attn.sum(dim=-1, keepdim=True)
        ent = attention_entropy(attn)
        self.assertLess(ent, math.log(N) * 0.5)


class TestComputeAllMetrics(unittest.TestCase):
    """Tests for the combined metrics function."""

    def test_returns_all_keys(self):
        """Should return all expected metric keys."""
        embeddings = torch.randn(4, 32, 128)
        attn = torch.softmax(torch.randn(4, 4, 32, 32), dim=-1)
        metrics = compute_all_metrics(embeddings, attn)
        expected_keys = [
            "effective_rank",
            "intrinsic_dim_mle",
            "anisotropy_index",
            "pairwise_dist_mean",
            "pairwise_dist_std",
            "pairwise_dist_min",
            "pairwise_dist_max",
            "pairwise_dist_median",
            "pairwise_dist_cv",
            "concentration_ratio",
            "attention_entropy",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_without_attention_matrix(self):
        """Should work without attention matrix (skip attention metrics)."""
        embeddings = torch.randn(4, 32, 128)
        metrics = compute_all_metrics(embeddings)
        self.assertIn("effective_rank", metrics)
        self.assertNotIn("concentration_ratio", metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
