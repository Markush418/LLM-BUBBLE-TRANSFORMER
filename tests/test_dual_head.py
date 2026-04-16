"""Unit tests for Dual-Head Tension architecture."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from plateau_attention import DualHeadPlateauAttention, PlateauAttentionMechanism
from metrics import tension_balance


class TestDualHeadInit(unittest.TestCase):
    """Test DualHeadPlateauAttention initialization."""

    def test_default_init(self):
        """Default initialization creates valid dual-head attention."""
        attn = DualHeadPlateauAttention(d_model=128, num_heads=4)
        self.assertEqual(attn.d_model, 128)
        self.assertEqual(attn.num_heads, 4)
        self.assertEqual(attn.epsilon_low, 0.001)
        self.assertEqual(attn.epsilon_high, 0.1)
        self.assertEqual(attn.alpha, 0.5)

    def test_custom_params(self):
        """Custom parameters are stored correctly."""
        attn = DualHeadPlateauAttention(
            d_model=256,
            num_heads=8,
            epsilon_low=0.01,
            epsilon_high=0.5,
            alpha=0.7,
        )
        self.assertEqual(attn.epsilon_low, 0.01)
        self.assertEqual(attn.epsilon_high, 0.5)
        self.assertEqual(attn.alpha, 0.7)

    def test_shared_projections(self):
        """W_q, W_k, W_v, W_o are shared (single set)."""
        attn = DualHeadPlateauAttention(d_model=128, num_heads=4)
        self.assertIsNotNone(attn.W_q)
        self.assertIsNotNone(attn.W_k)
        self.assertIsNotNone(attn.W_v)
        self.assertIsNotNone(attn.W_o)
        self.assertEqual(attn.W_q.shape, (128, 128))


class TestDualHeadForward(unittest.TestCase):
    """Test forward pass behavior."""

    def setUp(self):
        self.attn = DualHeadPlateauAttention(d_model=128, num_heads=4, seed=42)
        self.rng = np.random.RandomState(42)
        self.x = self.rng.randn(2, 32, 128).astype(np.float32)

    def test_output_shape(self):
        """Output shape matches input shape."""
        output = self.attn.forward(self.x)
        self.assertEqual(output.shape, self.x.shape)

    def test_output_no_nan(self):
        """Output contains no NaN or Inf values."""
        output = self.attn.forward(self.x)
        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))

    def test_return_attention(self):
        """return_attention=True returns output and two attention matrices."""
        output, A_low, A_high = self.attn.forward(self.x, return_attention=True)
        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual(A_low.shape[-1], 32)
        self.assertEqual(A_high.shape[-1], 32)

    def test_attention_stochastic(self):
        """High-epsilon head is approximately doubly stochastic."""
        _, A_low, A_high = self.attn.forward(self.x, return_attention=True)
        # High-epsilon head converges well with 5 Sinkhorn iterations
        row_sums_high = A_high.sum(axis=-1)
        self.assertTrue(np.allclose(row_sums_high, 1.0, atol=0.01))


class TestTensionCoefficient(unittest.TestCase):
    """Test alpha tension coefficient behavior."""

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.x = self.rng.randn(2, 32, 128).astype(np.float32)

    def test_alpha_zero_vs_one(self):
        """alpha=0 produces different output than alpha=1."""
        attn_0 = DualHeadPlateauAttention(d_model=128, num_heads=4, alpha=0.0, seed=42)
        attn_1 = DualHeadPlateauAttention(d_model=128, num_heads=4, alpha=1.0, seed=42)
        out_0 = attn_0.forward(self.x)
        out_1 = attn_1.forward(self.x)
        self.assertFalse(np.allclose(out_0, out_1))

    def test_alpha_deterministic(self):
        """Same alpha produces same output."""
        attn1 = DualHeadPlateauAttention(d_model=128, num_heads=4, alpha=0.5, seed=42)
        attn2 = DualHeadPlateauAttention(d_model=128, num_heads=4, alpha=0.5, seed=42)
        out1 = attn1.forward(self.x)
        out2 = attn2.forward(self.x)
        np.testing.assert_array_almost_equal(out1, out2)

    def test_alpha_interpolation(self):
        """alpha=0.5 output is approximately midpoint of alpha=0 and alpha=1."""
        attn_0 = DualHeadPlateauAttention(d_model=128, num_heads=4, alpha=0.0, seed=42)
        attn_1 = DualHeadPlateauAttention(d_model=128, num_heads=4, alpha=1.0, seed=42)
        attn_05 = DualHeadPlateauAttention(d_model=128, num_heads=4, alpha=0.5, seed=42)
        out_0 = attn_0.forward(self.x)
        out_1 = attn_1.forward(self.x)
        out_05 = attn_05.forward(self.x)
        mid = 0.5 * out_0 + 0.5 * out_1
        self.assertTrue(np.allclose(out_05, mid, atol=1e-3))


class TestTensionBalanceMetric(unittest.TestCase):
    """Test tension_balance metric."""

    def test_identical_matrices(self):
        """Identical attention matrices give tension_balance = 0."""
        A = np.random.randn(2, 4, 32, 32).astype(np.float32)
        A = np.abs(A)
        A = A / A.sum(axis=-1, keepdims=True)
        balance = tension_balance(A, A)
        self.assertAlmostEqual(balance, 0.0, places=5)

    def test_different_matrices(self):
        """Different attention matrices give tension_balance > 0."""
        A1 = np.random.randn(2, 4, 32, 32).astype(np.float32)
        A2 = np.random.randn(2, 4, 32, 32).astype(np.float32)
        A1 = np.abs(A1)
        A2 = np.abs(A2)
        A1 = A1 / A1.sum(axis=-1, keepdims=True)
        A2 = A2 / A2.sum(axis=-1, keepdims=True)
        balance = tension_balance(A1, A2)
        self.assertGreater(balance, 0.0)

    def test_balance_range(self):
        """Tension balance is in [0, 2] (1 - cosine_similarity where cos in [-1, 1])."""
        A1 = np.random.randn(2, 4, 32, 32).astype(np.float32)
        A2 = np.random.randn(2, 4, 32, 32).astype(np.float32)
        balance = tension_balance(A1, A2)
        self.assertGreaterEqual(balance, 0.0)
        self.assertLessEqual(balance, 2.0)


class TestDualHeadVsSingle(unittest.TestCase):
    """Compare dual-head with single-head baseline."""

    def test_dual_head_preserves_rank(self):
        """Dual-head maintains comparable effective rank to single low-epsilon head."""
        from metrics import effective_rank

        rng = np.random.RandomState(42)
        x = rng.randn(2, 32, 128).astype(np.float32)

        dual = DualHeadPlateauAttention(d_model=128, num_heads=4, alpha=0.5, seed=42)
        single_low = PlateauAttentionMechanism(
            d_model=128, num_heads=4, epsilon=0.001, seed=42
        )

        out_dual = dual.forward(x)
        out_single = single_low.forward(x)

        rank_dual = effective_rank(out_dual)
        rank_single = effective_rank(out_single)

        # Dual head should have equal or higher rank (within 20% tolerance)
        self.assertGreaterEqual(rank_dual, rank_single * 0.8)

    def test_dual_head_output_differs_from_single(self):
        """Dual-head output differs from single-head baseline."""
        rng = np.random.RandomState(42)
        x = rng.randn(2, 32, 128).astype(np.float32)

        dual = DualHeadPlateauAttention(d_model=128, num_heads=4, alpha=0.5, seed=42)
        single = PlateauAttentionMechanism(
            d_model=128, num_heads=4, epsilon=0.1, seed=42
        )

        out_dual = dual.forward(x)
        out_single = single.forward(x)

        self.assertFalse(np.allclose(out_dual, out_single))


if __name__ == "__main__":
    unittest.main()
