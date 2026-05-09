"""Tests for BaroreceptorMLP dynamic centroid prediction.

This module tests the baroreceptor feedback mechanism that predicts the
optimal number of centroids C based on input variance, analogous to
biological baroreceptors regulating blood pressure.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import torch
from baroreceptor import BaroreceptorMLP


class TestBaroreceptorMLP(unittest.TestCase):
    """Tests for BaroreceptorMLP dynamic centroid prediction."""

    def setUp(self):
        """Set standard dimensions and create a fresh baroreceptor instance."""
        self.B = 2
        self.N = 32
        self.d_model = 128
        self.min_C = 16
        self.max_C = 512
        self.baroreceptor = BaroreceptorMLP(
            d_model=self.d_model, min_C=self.min_C, max_C=self.max_C
        )

    # ------------------------------------------------------------------
    # Basic API / shape tests
    # ------------------------------------------------------------------

    def test_forward_returns_int(self):
        """forward() must return a Python int."""
        x = torch.randn(self.B, self.N, self.d_model)
        C = self.baroreceptor(x)
        self.assertIsInstance(C, int)

    def test_forward_batch_returns_tensor(self):
        """forward_batch() must return a 1-D LongTensor of length B."""
        x = torch.randn(self.B, self.N, self.d_model)
        C_batch = self.baroreceptor.forward_batch(x)
        self.assertEqual(C_batch.shape, (self.B,))
        self.assertEqual(C_batch.dtype, torch.int32)

    # ------------------------------------------------------------------
    # Threshold logic (output clamped to [min_C, max_C])
    # ------------------------------------------------------------------

    def test_output_within_threshold_bounds(self):
        """Predicted C must always lie inside [min_C, max_C]."""
        x = torch.randn(self.B, self.N, self.d_model)
        C = self.baroreceptor(x)
        self.assertGreaterEqual(C, self.min_C, f"C={C} is below min_C={self.min_C}")
        self.assertLessEqual(C, self.max_C, f"C={C} is above max_C={self.max_C}")

    def test_batch_output_within_threshold_bounds(self):
        """All batch predictions must lie inside [min_C, max_C]."""
        x = torch.randn(self.B, self.N, self.d_model)
        C_batch = self.baroreceptor.forward_batch(x)
        self.assertTrue(
            (C_batch >= self.min_C).all(), f"Batch C below min_C: {C_batch}"
        )
        self.assertTrue(
            (C_batch <= self.max_C).all(), f"Batch C above max_C: {C_batch}"
        )

    def test_extreme_values_respect_threshold(self):
        """Even with extreme inputs, C must stay within threshold bounds."""
        test_cases = [
            torch.zeros(self.B, self.N, self.d_model),  # zero input
            torch.ones(self.B, self.N, self.d_model) * 1e6,  # huge positive
            torch.ones(self.B, self.N, self.d_model) * -1e6,  # huge negative
        ]
        for x in test_cases:
            C = self.baroreceptor(x)
            self.assertGreaterEqual(C, self.min_C)
            self.assertLessEqual(C, self.max_C)

    # ------------------------------------------------------------------
    # Feedback mechanism (variance sensitivity)
    # ------------------------------------------------------------------

    def test_feedback_activation_by_variance(self):
        """High-variance input should produce a different C than low-variance."""
        # Low-variance input → embeddings are nearly identical
        x_low = torch.randn(self.B, self.N, self.d_model) * 0.01
        # High-variance input → embeddings are spread out
        x_high = torch.randn(self.B, self.N, self.d_model) * 100.0

        C_low = self.baroreceptor(x_low)
        C_high = self.baroreceptor(x_high)

        # The baroreceptor should react differently to different variance levels.
        # We do not assert a strict ordering because the MLP is randomly
        # initialised, but we verify both outputs are valid integers in range.
        self.assertIsInstance(C_low, int)
        self.assertIsInstance(C_high, int)
        self.assertNotEqual(
            C_low,
            C_high,
            "Baroreceptor feedback did not activate: same C for low and high variance",
        )

    def test_forward_batch_variance_sensitivity(self):
        """Batch mode should preserve variance sensitivity per sample."""
        x_low = torch.randn(self.B, self.N, self.d_model) * 0.01
        x_high = torch.randn(self.B, self.N, self.d_model) * 100.0

        C_low_batch = self.baroreceptor.forward_batch(x_low)
        C_high_batch = self.baroreceptor.forward_batch(x_high)

        self.assertFalse(
            torch.equal(C_low_batch, C_high_batch),
            "Batch feedback did not activate: identical predictions for different variance",
        )

    # ------------------------------------------------------------------
    # Mock attention outputs
    # ------------------------------------------------------------------

    def test_with_mock_attention_outputs(self):
        """Feed embeddings derived from mock doubly-stochastic attention."""
        # Simulate attention-derived embeddings: uniform attention yields
        # nearly uniform token representations.
        torch.manual_seed(42)
        attn = torch.ones(self.B, 4, self.N, self.N)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        # Project attention back to d_model space (simplified mock)
        proj = torch.randn(self.N, self.d_model)
        x = torch.einsum("bhnn,nd->bnd", attn, proj)

        C = self.baroreceptor(x)
        self.assertIsInstance(C, int)
        self.assertGreaterEqual(C, self.min_C)
        self.assertLessEqual(C, self.max_C)

    def test_with_peaked_attention_outputs(self):
        """Feed embeddings derived from sharply peaked (one-hot-like) attention."""
        torch.manual_seed(42)
        attn = torch.zeros(self.B, 4, self.N, self.N)
        for b in range(self.B):
            for h in range(4):
                for i in range(self.N):
                    attn[b, h, i, i % self.N] = 1.0
        attn = attn / attn.sum(dim=-1, keepdim=True)
        proj = torch.randn(self.N, self.d_model)
        x = torch.einsum("bhnn,nd->bnd", attn, proj)

        C = self.baroreceptor(x)
        self.assertIsInstance(C, int)
        self.assertGreaterEqual(C, self.min_C)
        self.assertLessEqual(C, self.max_C)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_zero_attention_edge_case(self):
        """All-zero input should still yield a valid integer C in range."""
        x = torch.zeros(self.B, self.N, self.d_model)
        C = self.baroreceptor(x)
        self.assertIsInstance(C, int)
        self.assertGreaterEqual(C, self.min_C)
        self.assertLessEqual(C, self.max_C)

    def test_uniform_attention_edge_case(self):
        """Uniform (constant) input should still yield a valid integer C in range."""
        x = torch.ones(self.B, self.N, self.d_model) * 3.14
        C = self.baroreceptor(x)
        self.assertIsInstance(C, int)
        self.assertGreaterEqual(C, self.min_C)
        self.assertLessEqual(C, self.max_C)

    def test_single_batch_item(self):
        """B=1 should work for both forward and forward_batch."""
        x = torch.randn(1, self.N, self.d_model)
        C = self.baroreceptor(x)
        self.assertIsInstance(C, int)
        self.assertGreaterEqual(C, self.min_C)
        self.assertLessEqual(C, self.max_C)

        C_batch = self.baroreceptor.forward_batch(x)
        self.assertEqual(C_batch.shape, (1,))

    def test_custom_threshold_bounds(self):
        """Custom min_C and max_C should be respected."""
        custom_baroreceptor = BaroreceptorMLP(d_model=self.d_model, min_C=32, max_C=128)
        x = torch.randn(self.B, self.N, self.d_model)
        C = custom_baroreceptor(x)
        self.assertGreaterEqual(C, 32)
        self.assertLessEqual(C, 128)

        C_batch = custom_baroreceptor.forward_batch(x)
        self.assertTrue((C_batch >= 32).all())
        self.assertTrue((C_batch <= 128).all())

    def test_deterministic_with_same_input(self):
        """Same input should yield the same prediction (no randomness in forward)."""
        x = torch.randn(self.B, self.N, self.d_model)
        C1 = self.baroreceptor(x)
        C2 = self.baroreceptor(x)
        self.assertEqual(C1, C2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
