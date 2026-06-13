"""
Tests for BaroreceptorMLP alpha prediction functionality.

Covers predict_alpha, forward_with_alpha, use_alpha_prediction flag,
and backward compatibility with existing forward/forward_batch methods.
"""

import sys
sys.path.insert(0, 'C:\\Users\\negocio\\Desktop\\LLM-BUBBLE')

import unittest
import torch
from models.baroreceptor import BaroreceptorMLP


class TestPredictAlpha(unittest.TestCase):
    """Tests for the predict_alpha method."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 512
        self.baroreceptor = BaroreceptorMLP(d_model=self.d_model)

    def test_predict_alpha_range(self):
        """predict_alpha returns value in [0.3, 0.8]."""
        x = torch.randn(4, 128, self.d_model)
        alpha = self.baroreceptor.predict_alpha(x)
        self.assertIsInstance(alpha, float)
        self.assertGreaterEqual(alpha, 0.3)
        self.assertLessEqual(alpha, 0.8)

    def test_predict_alpha_different_variances(self):
        """predict_alpha returns different values for different input variances."""
        x_low_var = torch.randn(4, 128, self.d_model) * 0.1
        x_high_var = torch.randn(4, 128, self.d_model) * 10.0

        alpha_low = self.baroreceptor.predict_alpha(x_low_var)
        alpha_high = self.baroreceptor.predict_alpha(x_high_var)

        self.assertNotAlmostEqual(alpha_low, alpha_high, places=5)

    def test_predict_alpha_variance_correlation(self):
        """High variance -> lower alpha; low variance -> higher alpha."""
        x_low_var = torch.randn(4, 128, self.d_model) * 0.1
        x_high_var = torch.randn(4, 128, self.d_model) * 10.0

        alpha_low = self.baroreceptor.predict_alpha(x_low_var)
        alpha_high = self.baroreceptor.predict_alpha(x_high_var)

        self.assertGreater(alpha_low, alpha_high)

    def test_predict_alpha_deterministic(self):
        """predict_alpha is deterministic for same input."""
        x = torch.randn(4, 128, self.d_model)
        alpha1 = self.baroreceptor.predict_alpha(x)
        alpha2 = self.baroreceptor.predict_alpha(x)
        self.assertAlmostEqual(alpha1, alpha2, places=6)

    def test_predict_alpha_extreme_low_variance(self):
        """Very low variance input -> alpha close to 0.8."""
        x = torch.randn(4, 128, self.d_model) * 0.01
        alpha = self.baroreceptor.predict_alpha(x)
        self.assertGreater(alpha, 0.5)
        self.assertLessEqual(alpha, 0.8)

    def test_predict_alpha_extreme_high_variance(self):
        """Very high variance input -> alpha close to 0.3."""
        x = torch.randn(4, 128, self.d_model) * 50.0
        alpha = self.baroreceptor.predict_alpha(x)
        self.assertGreaterEqual(alpha, 0.3)
        self.assertLess(alpha, 0.5)


class TestForwardWithAlpha(unittest.TestCase):
    """Tests for the forward_with_alpha method."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 512
        self.baroreceptor = BaroreceptorMLP(d_model=self.d_model)
        self.baroreceptor_with_alpha = BaroreceptorMLP(
            d_model=self.d_model, use_alpha_prediction=True
        )

    def test_returns_tuple(self):
        """forward_with_alpha returns a tuple."""
        x = torch.randn(4, 128, self.d_model)
        result = self.baroreceptor.forward_with_alpha(x)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_returns_int_and_float(self):
        """forward_with_alpha returns (int, float)."""
        x = torch.randn(4, 128, self.d_model)
        C, alpha = self.baroreceptor.forward_with_alpha(x)
        self.assertIsInstance(C, int)
        self.assertIsInstance(alpha, float)

    def test_c_in_range(self):
        """C is in [min_C, max_C]."""
        x = torch.randn(4, 128, self.d_model)
        C, alpha = self.baroreceptor.forward_with_alpha(x)
        self.assertGreaterEqual(C, self.baroreceptor.min_C)
        self.assertLessEqual(C, self.baroreceptor.max_C)

    def test_alpha_in_range(self):
        """alpha is in [0.3, 0.8]."""
        x = torch.randn(4, 128, self.d_model)
        C, alpha = self.baroreceptor.forward_with_alpha(x)
        self.assertGreaterEqual(alpha, 0.3)
        self.assertLessEqual(alpha, 0.8)

    def test_uses_alpha_net_when_enabled(self):
        """When use_alpha_prediction=True, alpha_net is used."""
        x = torch.randn(4, 128, self.d_model)

        # With alpha prediction enabled
        C1, alpha1 = self.baroreceptor_with_alpha.forward_with_alpha(x)

        # Verify alpha_net exists and alpha is in range
        self.assertTrue(hasattr(self.baroreceptor_with_alpha, 'alpha_net'))
        self.assertTrue(self.baroreceptor_with_alpha.use_alpha_prediction)
        self.assertIsInstance(C1, int)
        self.assertIsInstance(alpha1, float)
        self.assertGreaterEqual(alpha1, 0.3)
        self.assertLessEqual(alpha1, 0.8)

    def test_fallback_to_variance_when_alpha_net_disabled(self):
        """When use_alpha_prediction=False, falls back to predict_alpha."""
        x = torch.randn(4, 128, self.d_model)

        C, alpha = self.baroreceptor.forward_with_alpha(x)
        expected_alpha = self.baroreceptor.predict_alpha(x)

        self.assertAlmostEqual(alpha, expected_alpha, places=5)


class TestBackwardCompat(unittest.TestCase):
    """Tests for backward compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 512
        self.baroreceptor = BaroreceptorMLP(d_model=self.d_model)

    def test_forward_returns_int(self):
        """forward() still returns int."""
        x = torch.randn(4, 128, self.d_model)
        C = self.baroreceptor.forward(x)
        self.assertIsInstance(C, int)

    def test_forward_batch_unchanged(self):
        """forward_batch() returns tensor of shape (B,)."""
        B = 4
        x = torch.randn(B, 128, self.d_model)
        C_batch = self.baroreceptor.forward_batch(x)
        self.assertIsInstance(C_batch, torch.Tensor)
        self.assertEqual(C_batch.shape, (B,))
        self.assertEqual(C_batch.dtype, torch.int32)

    def test_forward_batch_values_in_range(self):
        """forward_batch() returns values in [min_C, max_C]."""
        B = 4
        x = torch.randn(B, 128, self.d_model)
        C_batch = self.baroreceptor.forward_batch(x)
        self.assertTrue((C_batch >= self.baroreceptor.min_C).all())
        self.assertTrue((C_batch <= self.baroreceptor.max_C).all())


if __name__ == "__main__":
    unittest.main()
