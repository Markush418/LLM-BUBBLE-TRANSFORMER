"""Tests for SDOTAttention (V3) \u2014 Bubble Transformer V3
====================================================

Unit tests for the original SDOTAttention module (V3), including:
- Constructor and initialization
- Forward pass output shape
- Attention properties (block-masked softmax within bubbles)
- Numerical stability
- Baroreceptor dynamic C
- Comparison with PlateauAttention patterns
"""

import sys
import os
import unittest
import time
import torch
import torch.nn.functional as F

# Required by task: add models/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
# Also add experiments/ for PlateauAttention and test_helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from sdot_attention import SDOTAttention
from v3_core import block_masked_attention, cluster_keys, voronoi_assign

# Robust import for test_helpers (works both direct and via python -m unittest)
try:
    from test_helpers import create_mock_embeddings
except ImportError:
    from tests.test_helpers import create_mock_embeddings


class TestSDOTAttentionV3Init(unittest.TestCase):
    """Tests for SDOTAttention V3 constructor."""

    def test_constructor_defaults(self):
        """Default parameters should initialize correctly."""
        d_model, num_heads = 512, 8
        module = SDOTAttention(d_model=d_model, num_heads=num_heads)
        self.assertEqual(module.d_model, d_model)
        self.assertEqual(module.num_heads, num_heads)
        self.assertEqual(module.head_dim, d_model // num_heads)
        self.assertEqual(module.num_centroids, 32)
        self.assertTrue(module.use_baroreceptor)

    def test_constructor_invalid_d_model(self):
        """Should raise AssertionError if d_model not divisible by num_heads."""
        with self.assertRaises(AssertionError):
            SDOTAttention(d_model=512, num_heads=7)

    def test_constructor_fixed_c(self):
        """Should work with fixed C (no baroreceptor)."""
        module = SDOTAttention(
            d_model=512, num_heads=8, num_centroids=64, use_baroreceptor=False
        )
        self.assertEqual(module.num_centroids, 64)
        self.assertFalse(module.use_baroreceptor)

    def test_constructor_creates_projections(self):
        """Should create W_q, W_k, W_v, W_o projections."""
        module = SDOTAttention(d_model=512, num_heads=8)
        self.assertIsInstance(module.W_q, torch.nn.Linear)
        self.assertIsInstance(module.W_k, torch.nn.Linear)
        self.assertIsInstance(module.W_v, torch.nn.Linear)
        self.assertIsInstance(module.W_o, torch.nn.Linear)

    def test_constructor_baroreceptor(self):
        """Should create baroreceptor when use_baroreceptor=True."""
        module = SDOTAttention(d_model=512, num_heads=8, use_baroreceptor=True)
        self.assertTrue(hasattr(module, "baroreceptor"))

    def test_constructor_no_baroreceptor(self):
        """Should not create baroreceptor when use_baroreceptor=False."""
        module = SDOTAttention(d_model=512, num_heads=8, use_baroreceptor=False)
        self.assertFalse(hasattr(module, "baroreceptor"))


class TestSDOTAttentionV3Forward(unittest.TestCase):
    """Tests for SDOTAttention V3 forward pass."""

    def setUp(self):
        self.B, self.N, self.d_model = 2, 128, 512
        self.num_heads = 8
        self.x = create_mock_embeddings(self.B, self.N, self.d_model)

    def test_output_shape(self):
        """Output shape must match input [B, N, d_model]."""
        module = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )
        output, _ = module(self.x)
        self.assertEqual(output.shape, (self.B, self.N, self.d_model))

    def test_forward_no_nan_or_inf(self):
        """Output should not contain NaN or Inf."""
        module = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )
        output, _ = module(self.x)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf")

    def test_forward_gradient_flow(self):
        """Gradients should flow through the module."""
        x = create_mock_embeddings(self.B, self.N, self.d_model)
        x.requires_grad_(True)
        module = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )
        output, _ = module(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse((x.grad == 0).all())

    def test_forward_return_assignments(self):
        """Should return valid bubble assignments when requested."""
        module = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )
        output, assignments = module(self.x, return_assignments=True)
        self.assertIsNotNone(assignments)
        self.assertEqual(assignments.shape, (self.B, self.num_heads, self.N))
        # Assignments should be valid centroid indices
        self.assertTrue((assignments >= 0).all())
        self.assertTrue((assignments < 32).all())

    def test_forward_no_assignments(self):
        """Should return None for assignments when not requested."""
        module = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )
        output, assignments = module(self.x, return_assignments=False)
        self.assertIsNone(assignments)

    def test_forward_with_baroreceptor(self):
        """Should work with baroreceptor (dynamic C)."""
        module = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_baroreceptor=True,
            min_C=16,
            max_C=128,
        )
        output, _ = module(self.x)
        self.assertEqual(output.shape, (self.B, self.N, self.d_model))
        self.assertFalse(torch.isnan(output).any())

    def test_forward_with_fixed_C(self):
        """forward_with_fixed_C should produce correct shape."""
        module = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_baroreceptor=True,
            min_C=16,
            max_C=128,
        )
        output, _ = module.forward_with_fixed_C(self.x, C=64)
        self.assertEqual(output.shape, (self.B, self.N, self.d_model))

    def test_edge_case_n_lt_c(self):
        """Should handle N < C gracefully."""
        x = create_mock_embeddings(self.B, 10, self.d_model)
        module = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )
        output, _ = module(x)
        self.assertEqual(output.shape, (self.B, 10, self.d_model))
        self.assertFalse(torch.isnan(output).any())

    def test_edge_case_single_token(self):
        """Should handle N=1."""
        x = create_mock_embeddings(self.B, 1, self.d_model)
        module = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=8,
            use_baroreceptor=False,
        )
        output, _ = module(x)
        self.assertEqual(output.shape, (self.B, 1, self.d_model))
        self.assertFalse(torch.isnan(output).any())


class TestSDOTAttentionV3BlockMasked(unittest.TestCase):
    """Tests for block_masked_attention: valid probability distribution."""

    def setUp(self):
        self.B, self.H, self.N, self.d = 2, 4, 32, 64
        self.C = 8
        self.Q = torch.randn(self.B, self.H, self.N, self.d)
        self.K = torch.randn(self.B, self.H, self.N, self.d)
        self.V = torch.randn(self.B, self.H, self.N, self.d)

    def _compute_block_attention_weights(self, assignments):
        """Helper: recompute attention weights to validate row sums."""
        scale = 1.0 / (self.d**0.5)
        attn_scores = torch.matmul(self.Q, self.K.transpose(-2, -1)) * scale
        assignments_expanded_row = assignments.unsqueeze(-1)
        assignments_expanded_col = assignments.unsqueeze(-2)
        bubble_mask = assignments_expanded_row == assignments_expanded_col
        attn_scores = attn_scores.masked_fill(~bubble_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        return attn_weights

    def test_attention_row_sums_to_one(self):
        """Attention matrix rows within bubbles must sum to 1.0."""
        centroids = cluster_keys(self.K, num_centroids=self.C)
        assignments = voronoi_assign(self.Q, centroids)
        attn_weights = self._compute_block_attention_weights(assignments)
        row_sums = attn_weights.sum(dim=-1)
        self.assertTrue(
            torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5),
            "Attention row sums deviate from 1.0",
        )

    def test_attention_non_negative(self):
        """All attention weights must be >= 0."""
        centroids = cluster_keys(self.K, num_centroids=self.C)
        assignments = voronoi_assign(self.Q, centroids)
        attn_weights = self._compute_block_attention_weights(assignments)
        self.assertTrue(
            torch.all(attn_weights >= 0), "Attention weights contain negative values"
        )

    def test_block_masked_output_shape(self):
        """block_masked_attention output shape must be [B, H, N, d]."""
        centroids = cluster_keys(self.K, num_centroids=self.C)
        assignments = voronoi_assign(self.Q, centroids)
        output = block_masked_attention(self.Q, self.K, self.V, assignments, centroids)
        self.assertEqual(output.shape, (self.B, self.H, self.N, self.d))

    def test_block_masked_numerical_stability(self):
        """Must not produce NaN or Inf with normal inputs."""
        centroids = cluster_keys(self.K, num_centroids=self.C)
        assignments = voronoi_assign(self.Q, centroids)
        output = block_masked_attention(self.Q, self.K, self.V, assignments, centroids)
        self.assertFalse(
            torch.isnan(output).any(), "Block attention output contains NaN"
        )
        self.assertFalse(
            torch.isinf(output).any(), "Block attention output contains Inf"
        )

    def test_empty_bubble_handling(self):
        """Should not crash when some centroids have no assigned tokens."""
        # Force a scenario where C > N by using small N
        Q = torch.randn(1, 1, 4, self.d)
        K = torch.randn(1, 1, 4, self.d)
        V = torch.randn(1, 1, 4, self.d)
        centroids = cluster_keys(K, num_centroids=8)
        assignments = voronoi_assign(Q, centroids)
        output = block_masked_attention(Q, K, V, assignments, centroids)
        self.assertEqual(output.shape, (1, 1, 4, self.d))
        self.assertFalse(torch.isnan(output).any())


class TestSDOTV3VersusPlateauPatterns(unittest.TestCase):
    """Compare SDOT V3 behavior with PlateauAttention patterns."""

    def setUp(self):
        self.B, self.N, self.d_model = 2, 32, 128
        self.num_heads = 4
        self.x = create_mock_embeddings(self.B, self.N, self.d_model)

    def test_same_output_shape_as_plateau(self):
        """Both SDOT V3 and PlateauAttention should emit [B, N, d_model]."""
        from plateau_attention import PlateauAttentionMechanism

        plateau = PlateauAttentionMechanism(
            d_model=self.d_model, num_heads=self.num_heads, epsilon=0.1
        )
        sdot = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=16,
            use_baroreceptor=False,
        )
        # PlateauAttentionMechanism is NumPy-only and uses .forward()
        x_np = self.x.detach().cpu().numpy()
        out_plateau = plateau.forward(x_np)
        out_sdot, _ = sdot(self.x)
        self.assertEqual(out_plateau.shape, out_sdot.shape)

    def test_sdot_completes_in_reasonable_time(self):
        """SDOT forward pass should complete quickly (smoke test for efficiency)."""
        sdot = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=16,
            use_baroreceptor=False,
        )
        # Warm-up
        for _ in range(3):
            _ = sdot(self.x)
        # Timed run
        start = time.time()
        for _ in range(10):
            _ = sdot(self.x)
        elapsed = time.time() - start
        self.assertLess(elapsed, 2.0, "SDOT forward pass too slow (>2s for 10 iters)")

    def test_numerical_stability_across_runs(self):
        """Multiple forward passes must remain numerically stable."""
        sdot = SDOTAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=16,
            use_baroreceptor=False,
        )
        for _ in range(5):
            x = create_mock_embeddings(self.B, self.N, self.d_model)
            output, _ = sdot(x)
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
