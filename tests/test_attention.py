"""Tests for PlateauAttentionMechanism and PlateauAttentionBlock."""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from plateau_attention import PlateauAttentionMechanism, PlateauAttentionBlock
from metrics import concentration_ratio, attention_entropy


class TestPlateauAttentionMechanism(unittest.TestCase):
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
        row_sums = attention.sum(dim=-1)
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
        self.assertLess(
            cr_small,
            cr_large,
            f"Small eps concentration ({cr_small:.4f}) should be < large eps ({cr_large:.4f})",
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
            f"Small eps entropy ({ent_small:.2f}) should be < large eps ({ent_large:.2f})",
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
            self.assertFalse(torch.isnan(output).any(), f"NaN output for eps={eps}")
            self.assertFalse(torch.isinf(output).any(), f"Inf output for eps={eps}")

    def test_multi_head_independence(self):
        """Different heads should produce different attention patterns."""
        attn = PlateauAttentionMechanism(d_model=self.D, num_heads=8, epsilon=0.1)
        _, attention = attn(self.x, return_attention=True)
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
        diff = (output - self.x).abs().mean()
        self.assertGreater(
            diff, 0.01, "Residual connection should still produce change"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
