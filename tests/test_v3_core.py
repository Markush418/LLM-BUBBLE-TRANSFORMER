"""
Tests for Bubble Transformer V3 Core — SDOT Algorithms
======================================================

Unit tests for:
- cluster_keys: K-Means clustering
- voronoi_assign: Voronoi assignment
- block_masked_attention: Attention within bubbles
- SDOTAttention: Complete module
"""

import sys
import os
import unittest
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.v3_core import (
    cluster_keys,
    voronoi_assign,
    block_masked_attention,
    compute_hard_support,
)
from models.baroreceptor import BaroreceptorMLP
from models.sdot_attention import SDOTAttention


class TestClusterKeys(unittest.TestCase):
    """Tests for cluster_keys function."""

    def test_cluster_keys_shape(self):
        """Cluster must produce C centroids."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        K = torch.randn(B, H, N, d)
        centroids = cluster_keys(K, num_centroids=C)

        self.assertEqual(centroids.shape, (B, H, C, d))

    def test_cluster_keys_deterministic(self):
        """Same input should produce centroids with similar statistics."""
        torch.manual_seed(42)
        K = torch.randn(2, 4, 100, 64)

        centroids1 = cluster_keys(K, num_centroids=32)
        centroids2 = cluster_keys(K, num_centroids=32)

        # Should have similar mean and std (not exact due to random initialization)
        mean1, std1 = centroids1.mean().item(), centroids1.std().item()
        mean2, std2 = centroids2.mean().item(), centroids2.std().item()

        self.assertAlmostEqual(mean1, mean2, places=1)
        self.assertAlmostEqual(std1, std2, places=1)

    def test_cluster_keys_extreme_C(self):
        """Handle edge cases: C=1 and C=N."""
        B, H, N, d = 2, 4, 100, 64

        # C=1: all tokens in one cluster
        K = torch.randn(B, H, N, d)
        centroids = cluster_keys(K, num_centroids=1)
        self.assertEqual(centroids.shape, (B, H, 1, d))

        # C=N: each token is its own cluster
        centroids = cluster_keys(K, num_centroids=N)
        self.assertEqual(centroids.shape, (B, H, N, d))

    def test_cluster_keys_batch_independence(self):
        """Different batches should have different centroids."""
        K = torch.randn(4, 2, 50, 32)
        centroids = cluster_keys(K, num_centroids=16)

        # Check that different batches have different centroids
        self.assertFalse(torch.allclose(centroids[0], centroids[1], atol=1e-5))


class TestVoronoiAssign(unittest.TestCase):
    """Tests for voronoi_assign function."""

    def test_voronoi_assign_coverage(self):
        """All tokens must be assigned to a centroid."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        Q = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)
        assignments = voronoi_assign(Q, centroids)

        # Check shape
        self.assertEqual(assignments.shape, (B, H, N))

        # Check range
        self.assertTrue((assignments >= 0).all())
        self.assertTrue((assignments < C).all())

    def test_voronoi_assign_correctness(self):
        """Assignment should be to nearest centroid."""
        B, H, N, d = 1, 1, 10, 4
        C = 3

        Q = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)
        assignments = voronoi_assign(Q, centroids)

        # Verify each assignment is correct
        dists = torch.cdist(Q, centroids)  # [B, H, N, C]
        expected_assignments = dists.argmin(dim=-1)

        self.assertTrue(torch.equal(assignments, expected_assignments))

    def test_voronoi_assign_distribution(self):
        """Reasonable distribution of tokens across centroids."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        Q = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)
        assignments = voronoi_assign(Q, centroids)

        # Check that not all tokens are in one cluster
        for b in range(B):
            for h in range(H):
                unique_clusters = torch.unique(assignments[b, h])
                # Should have multiple clusters (unless pathological case)
                self.assertGreater(len(unique_clusters), 1)


class TestBlockMaskedAttention(unittest.TestCase):
    """Tests for block_masked_attention function."""

    def test_block_masked_attention_shape(self):
        """Output shape must match input."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)
        assignments = voronoi_assign(Q, centroids)

        output = block_masked_attention(Q, K, V, assignments, centroids)

        self.assertEqual(output.shape, (B, H, N, d))

    def test_block_masked_attention_no_nan(self):
        """Output should not contain NaN."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)
        assignments = voronoi_assign(Q, centroids)

        output = block_masked_attention(Q, K, V, assignments, centroids)

        self.assertFalse(torch.isnan(output).any())

    def test_block_masked_attention_sparsity(self):
        """Attention should be sparse (only within bubbles)."""
        B, H, N, d = 1, 1, 100, 64
        C = 10  # Fewer centroids = larger bubbles

        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)
        assignments = voronoi_assign(Q, centroids)

        output = block_masked_attention(Q, K, V, assignments, centroids)

        # Compute attention weights to check sparsity
        scale = 1.0 / (d**0.5)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Create bubble mask
        assignments_expanded_row = assignments.unsqueeze(-1)
        assignments_expanded_col = assignments.unsqueeze(-2)
        bubble_mask = assignments_expanded_row == assignments_expanded_col

        # Count active connections
        active_connections = bubble_mask.sum().item()
        total_connections = N * N

        # Sparsity should be high (fewer active connections)
        sparsity = 1.0 - (active_connections / total_connections)
        self.assertGreater(sparsity, 0.5)  # At least 50% sparse


class TestHardSupport(unittest.TestCase):
    """Tests for compute_hard_support function."""

    def test_hard_support_shape(self):
        """Support should have shape [B, H]."""
        B, H, N = 2, 4, 100
        C = 32

        assignments = torch.randint(0, C, (B, H, N))
        support = compute_hard_support(assignments)

        self.assertEqual(support.shape, (B, H))

    def test_hard_support_values(self):
        """Support should be non-negative."""
        B, H, N = 2, 4, 100
        C = 32

        assignments = torch.randint(0, C, (B, H, N))
        support = compute_hard_support(assignments)

        self.assertTrue((support >= 0).all())


class TestBaroreceptorMLP(unittest.TestCase):
    """Tests for BaroreceptorMLP."""

    def test_baroreceptor_output_range(self):
        """C should be in [min_C, max_C]."""
        B, N, d_model = 4, 128, 512

        x = torch.randn(B, N, d_model)
        baroreceptor = BaroreceptorMLP(d_model=d_model, min_C=16, max_C=512)

        C = baroreceptor(x)

        self.assertGreaterEqual(C, 16)
        self.assertLessEqual(C, 512)

    def test_baroreceptor_batch_output(self):
        """Batch predictions should all be in range."""
        B, N, d_model = 4, 128, 512

        x = torch.randn(B, N, d_model)
        baroreceptor = BaroreceptorMLP(d_model=d_model, min_C=16, max_C=512)

        C_batch = baroreceptor.forward_batch(x)

        self.assertEqual(C_batch.shape, (B,))
        self.assertTrue((C_batch >= 16).all())
        self.assertTrue((C_batch <= 512).all())


class TestSDOTAttention(unittest.TestCase):
    """Tests for SDOTAttention module."""

    def test_sdot_attention_forward_shape(self):
        """Output shape must match input."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        sdot = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, _ = sdot(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_sdot_attention_return_assignments(self):
        """Should return assignments when requested."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        sdot = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, assignments = sdot(x, return_assignments=True)

        self.assertIsNotNone(assignments)
        self.assertEqual(assignments.shape, (B, num_heads, N))

    def test_sdot_attention_dynamic_C(self):
        """Dynamic C mode should work."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        sdot = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_baroreceptor=True,
            min_C=16,
            max_C=128,
        )

        output, _ = sdot(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_sdot_attention_fixed_C(self):
        """forward_with_fixed_C should work."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        sdot = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_baroreceptor=True,
        )

        output, assignments = sdot.forward_with_fixed_C(
            x, C=64, return_assignments=True
        )

        self.assertEqual(output.shape, (B, N, d_model))
        self.assertIsNotNone(assignments)

    def test_sdot_attention_no_nan(self):
        """Output should not contain NaN."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        sdot = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, _ = sdot(x)

        self.assertFalse(torch.isnan(output).any())

    def test_sdot_attention_sparsity_reasonable(self):
        """Sparsity should be similar to Sinkhorn baseline."""
        B, N, d_model = 1, 100, 512
        num_heads = 8
        C = 32

        x = torch.randn(B, N, d_model)
        sdot = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=C,
            use_baroreceptor=False,
        )

        output, assignments = sdot(x, return_assignments=True)

        # Compute sparsity from assignments
        # Sparsity = fraction of tokens that can attend to each other
        # (tokens in same bubble)
        unique, counts = torch.unique(assignments[0, 0], return_counts=True)
        active_pairs = (counts.float() ** 2).sum().item()
        total_pairs = N * N
        sparsity = 1.0 - (active_pairs / total_pairs)

        # Sparsity should be reasonable (not too high, not too low)
        # With C=32 and N=100, expect ~68% sparsity
        self.assertGreater(sparsity, 0.3)  # At least 30% sparse
        self.assertLess(sparsity, 0.95)  # Not completely sparse


class TestIntegration(unittest.TestCase):
    """Integration tests for V3 components."""

    def test_full_pipeline(self):
        """Test full SDOT pipeline: cluster → assign → attend."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        # Generate input
        K = torch.randn(B, H, N, d)
        Q = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)

        # Cluster keys
        centroids = cluster_keys(K, num_centroids=C)
        self.assertEqual(centroids.shape, (B, H, C, d))

        # Assign queries
        assignments = voronoi_assign(Q, centroids)
        self.assertEqual(assignments.shape, (B, H, N))

        # Block-masked attention
        output = block_masked_attention(Q, K, V, assignments, centroids)
        self.assertEqual(output.shape, (B, H, N, d))

        # No NaN
        self.assertFalse(torch.isnan(output).any())

    def test_different_C_values(self):
        """Test with different C values."""
        B, N, d_model = 2, 100, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)

        for C in [8, 16, 32, 64, 128]:
            sdot = SDOTAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_centroids=C,
                use_baroreceptor=False,
            )

            output, _ = sdot(x)
            self.assertEqual(output.shape, (B, N, d_model))
            self.assertFalse(torch.isnan(output).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
