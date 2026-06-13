"""Integration tests for DualHeadSDOTAttentionV4.

Tests cover:
- Initialization (default params, custom params, share_projections True/False)
- Forward pass (output shape, numerical stability, return_assignments)
- Alpha variation (alpha=0, alpha=1, alpha=0.5, deterministic)
- GQA compatibility (num_kv_heads < num_heads)
- Shared vs unshared projections
- Comparison with single-head SDOTAttentionV4
"""

import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from models.sdot_attention_v4 import DualHeadSDOTAttentionV4, SDOTAttentionV4


class TestDualHeadV4Init(unittest.TestCase):
    """Tests for DualHeadSDOTAttentionV4 constructor."""

    def setUp(self):
        """Set canonical seed for reproducibility."""
        torch.manual_seed(42)

    def test_init_defaults(self):
        """Default parameters should initialize correctly."""
        module = DualHeadSDOTAttentionV4(d_model=512, num_heads=8)

        self.assertEqual(module.head_low.d_model, 512)
        self.assertEqual(module.head_low.num_heads, 8)
        self.assertEqual(module.epsilon_low, 0.001)
        self.assertEqual(module.epsilon_high, 0.1)
        self.assertEqual(module.alpha, 0.5)
        self.assertTrue(module.share_projections)
        self.assertIsInstance(module.head_low, SDOTAttentionV4)
        self.assertIsInstance(module.head_high, SDOTAttentionV4)

    def test_init_custom_params(self):
        """Custom parameters should be stored correctly."""
        module = DualHeadSDOTAttentionV4(
            d_model=1024,
            num_heads=16,
            num_centroids=64,
            epsilon_low=0.01,
            epsilon_high=0.5,
            alpha=0.7,
            share_projections=False,
            use_baroreceptor=False,
        )

        self.assertEqual(module.head_low.d_model, 1024)
        self.assertEqual(module.head_low.num_heads, 16)
        self.assertEqual(module.epsilon_low, 0.01)
        self.assertEqual(module.epsilon_high, 0.5)
        self.assertEqual(module.alpha, 0.7)
        self.assertFalse(module.share_projections)

    def test_init_share_projections_true(self):
        """When share_projections=True, W_q/k/v/o parameters should be shared."""
        module = DualHeadSDOTAttentionV4(
            d_model=512, num_heads=8, share_projections=True
        )

        self.assertIs(module.head_low.W_q.weight, module.head_high.W_q.weight)
        self.assertIs(module.head_low.W_k.weight, module.head_high.W_k.weight)
        self.assertIs(module.head_low.W_v.weight, module.head_high.W_v.weight)
        self.assertIs(module.head_low.W_o.weight, module.head_high.W_o.weight)

    def test_init_share_projections_false(self):
        """When share_projections=False, W_q/k/v/o should be independent."""
        module = DualHeadSDOTAttentionV4(
            d_model=512, num_heads=8, share_projections=False
        )

        self.assertIsNot(module.head_low.W_q.weight, module.head_high.W_q.weight)
        self.assertIsNot(module.head_low.W_k.weight, module.head_high.W_k.weight)
        self.assertIsNot(module.head_low.W_v.weight, module.head_high.W_v.weight)
        self.assertIsNot(module.head_low.W_o.weight, module.head_high.W_o.weight)


class TestDualHeadV4Forward(unittest.TestCase):
    """Tests for DualHeadSDOTAttentionV4 forward pass."""

    def setUp(self):
        """Set canonical seed and standard dimensions."""
        torch.manual_seed(42)
        self.B = 2
        self.N = 128
        self.d_model = 512
        self.num_heads = 8

    def test_forward_output_shape(self):
        """Output shape must be [B, N, d_model]."""
        x = torch.randn(self.B, self.N, self.d_model)
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_baroreceptor=False,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (self.B, self.N, self.d_model))

    def test_forward_no_nan_or_inf(self):
        """Output should not contain NaN or Inf."""
        x = torch.randn(self.B, self.N, self.d_model)
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_baroreceptor=False,
        )

        output, _ = module(x)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_forward_return_assignments(self):
        """Should return assignments info when requested."""
        x = torch.randn(self.B, self.N, self.d_model)
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_baroreceptor=False,
        )

        output, assignments_info = module(x, return_assignments=True)

        self.assertIsNotNone(assignments_info)
        self.assertIn("head_low", assignments_info)
        self.assertIn("head_high", assignments_info)
        self.assertIn("alpha", assignments_info)
        self.assertEqual(assignments_info["alpha"], module.alpha)

    def test_forward_with_attention_mask(self):
        """Should handle attention_mask parameter."""
        x = torch.randn(self.B, self.N, self.d_model)
        attention_mask = torch.ones(self.B, self.N)
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_baroreceptor=False,
        )

        output, _ = module(x, attention_mask=attention_mask)

        self.assertEqual(output.shape, (self.B, self.N, self.d_model))

    def test_forward_with_previous_centroids(self):
        """Should accept previous_centroids warm-start."""
        x = torch.randn(self.B, self.N, self.d_model)
        head_dim = self.d_model // self.num_heads
        num_centroids = 32
        previous_centroids = torch.randn(
            self.B, self.num_heads, num_centroids, head_dim
        )
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_centroids=num_centroids,
            use_baroreceptor=False,
        )

        output, assignments = module(
            x, return_assignments=True, previous_centroids=previous_centroids
        )

        self.assertEqual(output.shape, (self.B, self.N, self.d_model))
        self.assertIsNotNone(assignments)


class TestDualHeadV4Alpha(unittest.TestCase):
    """Tests for alpha tension fusion."""

    def setUp(self):
        """Set canonical seed and standard dimensions."""
        torch.manual_seed(42)
        self.B = 2
        self.N = 128
        self.d_model = 512
        self.num_heads = 8

    def test_alpha_zero_pure_high_head(self):
        """alpha=0 should produce pure high-head output."""
        x = torch.randn(self.B, self.N, self.d_model)
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            alpha=0.0,
            share_projections=True,
            use_baroreceptor=False,
        )

        output, _ = module(x)
        out_low, _ = module.head_low(x)
        out_high, _ = module.head_high(x)

        expected = out_high
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_alpha_one_pure_low_head(self):
        """alpha=1 should produce pure low-head output."""
        x = torch.randn(self.B, self.N, self.d_model)
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            alpha=1.0,
            share_projections=True,
            use_baroreceptor=False,
        )

        output, _ = module(x)
        out_low, _ = module.head_low(x)

        expected = out_low
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_alpha_half_balanced(self):
        """alpha=0.5 should produce balanced fusion."""
        x = torch.randn(self.B, self.N, self.d_model)
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            alpha=0.5,
            share_projections=True,
            use_baroreceptor=False,
        )

        output, _ = module(x)
        out_low, _ = module.head_low(x)
        out_high, _ = module.head_high(x)

        expected = 0.5 * out_low + 0.5 * out_high
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_alpha_deterministic(self):
        """Same input with same seed should produce same output."""
        torch.manual_seed(42)
        x = torch.randn(self.B, self.N, self.d_model)
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            alpha=0.3,
            share_projections=True,
            use_baroreceptor=False,
        )

        output1, _ = module(x)

        torch.manual_seed(42)
        x2 = torch.randn(self.B, self.N, self.d_model)
        module2 = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            alpha=0.3,
            share_projections=True,
            use_baroreceptor=False,
        )

        output2, _ = module2(x2)

        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestDualHeadV4GQA(unittest.TestCase):
    """Tests for GQA (Grouped Query Attention) compatibility."""

    def setUp(self):
        """Set canonical seed for reproducibility."""
        torch.manual_seed(42)

    def test_gqa_num_kv_heads_less_than_num_heads(self):
        """Should accept num_kv_heads < num_heads."""
        module = DualHeadSDOTAttentionV4(
            d_model=1024,
            num_heads=16,
            num_kv_heads=8,
            use_baroreceptor=False,
        )

        self.assertEqual(module.head_low.num_kv_heads, 8)
        self.assertEqual(module.head_high.num_kv_heads, 8)

    def test_gqa_forward_pass(self):
        """Forward pass should work with GQA."""
        B, N, d_model = 2, 100, 1024
        num_heads = 16
        num_kv_heads = 8

        x = torch.randn(B, N, d_model)
        module = DualHeadSDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_baroreceptor=False,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_gqa_projection_shapes(self):
        """K, V projections should have correct shapes with GQA."""
        module = DualHeadSDOTAttentionV4(
            d_model=1024,
            num_heads=16,
            num_kv_heads=8,
            use_baroreceptor=False,
        )

        # W_k: [num_kv_heads * head_dim, d_model] = [512, 1024]
        self.assertEqual(module.head_low.W_k.weight.shape, (512, 1024))
        self.assertEqual(module.head_high.W_k.weight.shape, (512, 1024))
        # W_v: [num_kv_heads * head_dim, d_model] = [512, 1024]
        self.assertEqual(module.head_low.W_v.weight.shape, (512, 1024))
        self.assertEqual(module.head_high.W_v.weight.shape, (512, 1024))


class TestDualHeadV4VsSingle(unittest.TestCase):
    """Tests comparing DualHead with single-head SDOTAttentionV4."""

    def setUp(self):
        """Set canonical seed and standard dimensions."""
        torch.manual_seed(42)
        self.B = 2
        self.N = 128
        self.d_model = 512
        self.num_heads = 8

    def test_shared_projections_same_weights(self):
        """Shared projections should use identical parameter tensors."""
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            share_projections=True,
            use_baroreceptor=False,
        )

        # Modify head_low's W_q weight
        with torch.no_grad():
            original = module.head_low.W_q.weight.clone()
            module.head_low.W_q.weight.add_(1.0)

        # head_high should see the same change
        self.assertTrue(
            torch.allclose(module.head_high.W_q.weight, module.head_low.W_q.weight)
        )
        # Should differ from original
        self.assertFalse(
            torch.allclose(module.head_high.W_q.weight, original, atol=1e-6)
        )

    def test_unshared_projections_different_weights(self):
        """Unshared projections should have different random weights."""
        module = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            share_projections=False,
            use_baroreceptor=False,
        )

        # Weights should differ between the two heads
        self.assertFalse(
            torch.allclose(
                module.head_low.W_q.weight,
                module.head_high.W_q.weight,
                atol=1e-6,
            )
        )

    def test_output_shape_matches_single_head(self):
        """Dual-head output shape should match single-head output shape."""
        x = torch.randn(self.B, self.N, self.d_model)

        dual = DualHeadSDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_baroreceptor=False,
        )
        single = SDOTAttentionV4(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_baroreceptor=False,
        )

        out_dual, _ = dual(x)
        out_single, _ = single(x)

        self.assertEqual(out_dual.shape, out_single.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
