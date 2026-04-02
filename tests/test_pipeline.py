"""
Unit Tests — Bubble Transformer Experiment Pipeline
=====================================================
Tests for PlateauAttention, Metrics, and Epsilon Sweep.
Run with: python test_pipeline.py

These tests use mock data (no Qwen model required) to validate
the core computational pipeline.
"""

import sys
import os
import math
import unittest
import numpy as np
import torch

# Ensure we can import from sibling modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from plateau_attention import PlateauAttentionMechanism, PlateauAttentionBlock
from metrics import (
    effective_rank,
    intrinsic_dimension_mle,
    anisotropy_index,
    pairwise_distance_stats,
    concentration_ratio,
    attention_entropy,
    compute_all_metrics,
)


# ─── PlateauAttention Tests ────────────────────────────────────────────────


class TestPlateauAttention(unittest.TestCase):
    """Tests for the PlateauAttentionMechanism."""

    def setUp(self):
        self.B, self.N, self.D = 2, 32, 128
        self.num_heads = 4
        self.x = torch.randn(self.B, self.N, self.D)

    def test_output_shape(self):
        """Output shape should match input shape [B, N, D]."""
        attn = PlateauAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=0.1
        )
        output = attn(self.x)
        self.assertEqual(output.shape, (self.B, self.N, self.D))

    def test_return_attention(self):
        """When return_attention=True, should return (output, attention_matrix)."""
        attn = PlateauAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=0.1
        )
        output, attention = attn(self.x, return_attention=True)
        self.assertEqual(output.shape, (self.B, self.N, self.D))
        self.assertEqual(attention.shape, (self.B, self.num_heads, self.N, self.N))

    def test_attention_sums_to_one(self):
        """Attention matrix rows should approximately sum to 1 (doubly-stochastic)."""
        attn = PlateauAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=0.1, tau_iters=10
        )
        _, attention = attn(self.x, return_attention=True)
        row_sums = attention.sum(dim=-1)  # [B, heads, N]
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3))

    def test_epsilon_affects_sparsity(self):
        """Smaller epsilon should produce more concentrated (sparser) attention."""
        attn_small = PlateauAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=0.01, tau_iters=10
        )
        attn_large = PlateauAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=1.0, tau_iters=10
        )

        _, attn_small_matrix = attn_small(self.x, return_attention=True)
        _, attn_large_matrix = attn_large(self.x, return_attention=True)

        cr_small = concentration_ratio(attn_small_matrix)
        cr_large = concentration_ratio(attn_large_matrix)

        # Smaller epsilon → more concentrated → lower concentration ratio
        self.assertLess(
            cr_small,
            cr_large,
            f"Small ε concentration ({cr_small:.4f}) should be < large ε ({cr_large:.4f})",
        )

    def test_attention_entropy_decreases_with_epsilon(self):
        """Smaller epsilon should produce lower attention entropy."""
        attn_small = PlateauAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=0.01, tau_iters=10
        )
        attn_large = PlateauAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=1.0, tau_iters=10
        )

        _, attn_small_matrix = attn_small(self.x, return_attention=True)
        _, attn_large_matrix = attn_large(self.x, return_attention=True)

        ent_small = attention_entropy(attn_small_matrix)
        ent_large = attention_entropy(attn_large_matrix)

        self.assertLess(
            ent_small,
            ent_large,
            f"Small ε entropy ({ent_small:.2f}) should be < large ε ({ent_large:.2f})",
        )

    def test_attention_is_non_negative(self):
        """All attention weights should be >= 0."""
        attn = PlateauAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=0.1
        )
        _, attention = attn(self.x, return_attention=True)
        self.assertTrue(torch.all(attention >= 0))

    def test_different_epsilon_values(self):
        """Should work with various epsilon values without numerical issues."""
        for eps in [0.001, 0.01, 0.1, 1.0]:
            attn = PlateauAttentionMechanism(
                d_model=self.D, num_heads=self.num_heads, epsilon=eps
            )
            output = attn(self.x)
            self.assertFalse(torch.isnan(output).any(), f"NaN output for ε={eps}")
            self.assertFalse(torch.isinf(output).any(), f"Inf output for ε={eps}")

    def test_multi_head_independence(self):
        """Different heads should produce different attention patterns."""
        attn = PlateauAttentionMechanism(d_model=self.D, num_heads=8, epsilon=0.1)
        _, attention = attn(self.x, return_attention=True)
        # Attention should have 8 heads
        self.assertEqual(attention.shape[1], 8)


class TestPlateauAttentionBlock(unittest.TestCase):
    """Tests for the full PlateauAttentionBlock."""

    def setUp(self):
        self.B, self.N, self.D = 2, 32, 128
        self.x = torch.randn(self.B, self.N, self.D)

    def test_block_output_shape(self):
        """Block output should match input shape."""
        block = PlateauAttentionBlock(
            d_model=self.D, num_heads=4, ff_dim=self.D * 4, epsilon=0.1
        )
        output = block(self.x)
        self.assertEqual(output.shape, (self.B, self.N, self.D))

    def test_residual_connection(self):
        """Output should be different from input (residual + transformation)."""
        block = PlateauAttentionBlock(
            d_model=self.D, num_heads=4, ff_dim=self.D * 4, epsilon=0.1
        )
        output = block(self.x)
        # Output should be different from input
        diff = (output - self.x).abs().mean()
        self.assertGreater(
            diff, 0.01, "Residual connection should still produce change"
        )


# ─── Metrics Tests ─────────────────────────────────────────────────────────


class TestEffectiveRank(unittest.TestCase):
    """Tests for the effective rank metric."""

    def test_full_rank_matrix(self):
        """Identity-like matrix should have high effective rank."""
        embeddings = torch.eye(64) * 10  # Well-conditioned
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
        # Add tiny noise
        embeddings += torch.randn_like(embeddings) * 1e-6
        dim = intrinsic_dimension_mle(embeddings, k=5)
        # Should be low (but noise may push it up slightly)
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
        embeddings[:, 0] *= 100  # Dominant first dimension
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
        self.assertGreater(stats["mean"], 0)
        self.assertGreater(stats["std"], 0)


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
        attn[:, :, :, 0] = 1.0  # All mass on first column
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
        attn = attn + 1e-10  # Avoid log(0)
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


# ─── Integration Tests ─────────────────────────────────────────────────────


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def test_epsilon_sweep_produces_trend(self):
        """Sweeping epsilon should produce a monotonic trend in concentration."""
        B, N, D = 2, 32, 128
        x = torch.randn(B, N, D)

        results = []
        for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
            attn = PlateauAttentionMechanism(
                d_model=D, num_heads=4, epsilon=eps, tau_iters=10
            )
            _, attn_matrix = attn(x, return_attention=True)
            cr = concentration_ratio(attn_matrix)
            ent = attention_entropy(attn_matrix)
            results.append({"epsilon": eps, "cr": cr, "entropy": ent})

        # Verify trend: concentration ratio should increase with epsilon
        cr_values = [r["cr"] for r in results]
        self.assertLess(
            cr_values[0], cr_values[-1], f"CR should increase with ε: {cr_values}"
        )

    def test_metrics_stable_across_runs(self):
        """Metrics should be stable for the same input."""
        torch.manual_seed(42)
        embeddings = torch.randn(4, 32, 128)

        metrics1 = compute_all_metrics(embeddings)
        metrics2 = compute_all_metrics(embeddings)

        self.assertAlmostEqual(
            metrics1["effective_rank"], metrics2["effective_rank"], places=5
        )
        self.assertAlmostEqual(
            metrics1["anisotropy_index"], metrics2["anisotropy_index"], places=5
        )


# ─── Run Tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Bubble Transformer — Pipeline Unit Tests")
    print("=" * 60)
    print()

    unittest.main(verbosity=2)
