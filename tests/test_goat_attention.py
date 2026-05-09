"""Tests for GOATAttention — Gated Optimal Attention Transport.

This module covers the GOAT attention classes defined in
``experiments/goat_attention.py``:

* ``GOATAttentionMechanism`` – Plateau attention with learnable per-head gates.
* ``DualHeadGOATAttention`` – Two GOAT mechanisms fused via a tension coefficient.
* ``GOATCostFunction`` – Cost matrix wrapper that applies multiplicative gates.

Conventions follow the existing test suite (``tests/test_attention.py``):
fixtures are PyTorch tensors, converted to NumPy before calling the pure-NumPy
production code.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from experiments.goat_attention import (
    GOATAttentionMechanism,
    DualHeadGOATAttention,
    GOATCostFunction,
)
from test_helpers import create_mock_embeddings


class TestGOATAttentionMechanism(unittest.TestCase):
    """Tests for ``GOATAttentionMechanism``."""

    def setUp(self):
        """Standard dimensions + PyTorch fixture, converted to NumPy."""
        self.B, self.N, self.D = 2, 32, 128
        self.num_heads = 4
        self.x_torch = create_mock_embeddings(self.B, self.N, self.D)
        self.x = self.x_torch.detach().cpu().numpy().astype(np.float32)

    def test_initialization(self):
        """GOATAttentionMechanism should initialise with correct attributes."""
        goat = GOATAttentionMechanism(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon=0.1,
            learn_gates=True,
        )
        self.assertEqual(goat.d_model, self.D)
        self.assertEqual(goat.num_heads, self.num_heads)
        self.assertTrue(goat.learn_gates)
        self.assertTrue(hasattr(goat.cost_fn, "get_gates"))
        self.assertTrue(hasattr(goat.cost_fn, "set_gates"))

    def test_output_shape(self):
        """Output shape should match input shape ``[B, N, D]``."""
        goat = GOATAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=0.1
        )
        output = goat.forward(self.x)
        self.assertEqual(output.shape, (self.B, self.N, self.D))

    def test_gating_mechanism_scales_cost(self):
        """GOATCostFunction should scale the underlying cost matrix by gates."""
        from plateau_attention import L2SquaredCost

        base_cost = L2SquaredCost()
        goat_cost = GOATCostFunction(
            base_cost=base_cost, num_heads=self.num_heads, gate_init=1.0
        )
        # Force gates to exactly 1.0 so the baseline is clean
        goat_cost.set_gates(np.ones(self.num_heads, dtype=np.float32))

        B, N, head_dim = 2, 16, 8
        Q = np.random.randn(B, self.num_heads, N, head_dim).astype(np.float32)
        K = np.random.randn(B, self.num_heads, N, head_dim).astype(np.float32)

        C_default = goat_cost.compute(Q, K)

        # Scale all gates by 5
        goat_cost.set_gates(np.ones(self.num_heads, dtype=np.float32) * 5.0)
        C_high = goat_cost.compute(Q, K)

        # After scaling gates by 5, the cost matrix should be ~5x larger
        ratio = C_high / (C_default + 1e-10)
        np.testing.assert_allclose(
            ratio,
            5.0,
            atol=1e-5,
            err_msg="Cost matrix should scale linearly with gates",
        )

    def test_gate_clipping(self):
        """Gates must be clipped to the range ``[0.01, 10.0]``."""
        goat = GOATAttentionMechanism(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon=0.1,
            learn_gates=True,
        )
        extreme = np.array([-100.0, 0.0, 1.0, 50.0], dtype=np.float32)
        goat.set_gates(extreme)
        gates = goat.get_gates()
        self.assertTrue(
            np.all(gates >= 0.01),
            f"Gates below minimum detected: {gates}",
        )
        self.assertTrue(
            np.all(gates <= 10.0),
            f"Gates above maximum detected: {gates}",
        )

    def test_numerical_stability(self):
        """Output must not contain NaN or Inf values."""
        goat = GOATAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=0.1
        )
        output = goat.forward(self.x)
        self.assertFalse(
            np.isnan(output).any(),
            "Output contains NaN values",
        )
        self.assertFalse(
            np.isinf(output).any(),
            "Output contains Inf values",
        )

    def test_edge_case_small_epsilon(self):
        """Very small epsilon (high concentration) must remain numerically stable."""
        goat = GOATAttentionMechanism(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon=0.001,
            tau_iters=10,
        )
        output = goat.forward(self.x)
        self.assertEqual(output.shape, (self.B, self.N, self.D))
        self.assertFalse(
            np.isnan(output).any(),
            "NaN detected with small epsilon",
        )
        self.assertFalse(
            np.isinf(output).any(),
            "Inf detected with small epsilon",
        )

    def test_edge_case_large_epsilon(self):
        """Large epsilon (near-uniform attention) must remain numerically stable."""
        goat = GOATAttentionMechanism(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon=1.0,
            tau_iters=10,
        )
        output = goat.forward(self.x)
        self.assertEqual(output.shape, (self.B, self.N, self.D))
        self.assertFalse(
            np.isnan(output).any(),
            "NaN detected with large epsilon",
        )
        self.assertFalse(
            np.isinf(output).any(),
            "Inf detected with large epsilon",
        )

    def test_return_attention(self):
        """``return_attention=True`` should yield ``(output, attention_matrix)``."""
        goat = GOATAttentionMechanism(
            d_model=self.D, num_heads=self.num_heads, epsilon=0.1
        )
        output, attention = goat.forward(self.x, return_attention=True)
        self.assertEqual(output.shape, (self.B, self.N, self.D))
        self.assertEqual(
            attention.shape,
            (self.B, self.num_heads, self.N, self.N),
        )

    def test_attention_sums_to_one(self):
        """Attention matrix rows should approximately sum to 1 (doubly-stochastic)."""
        goat = GOATAttentionMechanism(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon=0.1,
            tau_iters=10,
        )
        _, attention = goat.forward(self.x, return_attention=True)
        row_sums = attention.sum(axis=-1)
        np.testing.assert_allclose(
            row_sums,
            np.ones_like(row_sums),
            atol=1e-3,
            err_msg="Attention rows do not sum to 1",
        )

    def test_learn_gates_false(self):
        """With ``learn_gates=False`` the base cost function should be used."""
        goat = GOATAttentionMechanism(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon=0.1,
            learn_gates=False,
        )
        output = goat.forward(self.x)
        self.assertEqual(output.shape, (self.B, self.N, self.D))
        # cost_fn should NOT be a GOATCostFunction
        self.assertFalse(
            hasattr(goat.cost_fn, "get_gates"),
            "Base cost function should not expose gates",
        )


class TestDualHeadGOATAttention(unittest.TestCase):
    """Tests for ``DualHeadGOATAttention``."""

    def setUp(self):
        self.B, self.N, self.D = 2, 32, 128
        self.num_heads = 4
        self.x_torch = create_mock_embeddings(self.B, self.N, self.D)
        self.x = self.x_torch.detach().cpu().numpy().astype(np.float32)

    def test_dual_head_output_shape(self):
        """DualHeadGOATAttention output should match input shape ``[B, N, D]``."""
        dual = DualHeadGOATAttention(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon_low=0.01,
            epsilon_high=0.5,
            alpha=0.5,
        )
        output = dual.forward(self.x)
        self.assertEqual(output.shape, (self.B, self.N, self.D))

    def test_alpha_fusion(self):
        """Changing ``alpha`` should blend the two heads differently."""
        dual = DualHeadGOATAttention(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon_low=0.01,
            epsilon_high=0.5,
            alpha=0.5,
        )
        out_05 = dual.forward(self.x)

        dual.alpha = 0.0
        out_0 = dual.forward(self.x)

        dual.alpha = 1.0
        out_1 = dual.forward(self.x)

        diff = np.mean(np.abs(out_0 - out_1))
        self.assertGreater(
            diff,
            1e-6,
            "Alpha extremes should produce different outputs",
        )

        # out_05 should be roughly the weighted average
        expected = 0.5 * out_1 + 0.5 * out_0
        np.testing.assert_allclose(
            out_05,
            expected,
            atol=1e-5,
            err_msg="Alpha=0.5 should equal the weighted average",
        )

    def test_dual_head_gates(self):
        """``get_gates`` should return two distinct gate vectors."""
        dual = DualHeadGOATAttention(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon_low=0.01,
            epsilon_high=0.5,
            alpha=0.5,
        )
        gates_low, gates_high = dual.get_gates()
        self.assertEqual(gates_low.shape, (self.num_heads,))
        self.assertEqual(gates_high.shape, (self.num_heads,))

    def test_dual_head_numerical_stability(self):
        """DualHeadGOATAttention must not produce NaN or Inf."""
        dual = DualHeadGOATAttention(
            d_model=self.D,
            num_heads=self.num_heads,
            epsilon_low=0.01,
            epsilon_high=0.5,
            alpha=0.5,
        )
        output = dual.forward(self.x)
        self.assertFalse(np.isnan(output).any(), "NaN in dual-head output")
        self.assertFalse(np.isinf(output).any(), "Inf in dual-head output")


class TestGOATCostFunction(unittest.TestCase):
    """Tests for ``GOATCostFunction``."""

    def setUp(self):
        self.num_heads = 4

    def test_initialization(self):
        """Gates should be initialised near ``gate_init`` and clipped."""
        cost_fn = GOATCostFunction(num_heads=self.num_heads, gate_init=1.0, seed=42)
        gates = cost_fn.get_gates()
        self.assertEqual(gates.shape, (self.num_heads,))
        self.assertTrue(np.all(gates >= 0.01))
        self.assertTrue(np.all(gates <= 10.0))

    def test_gate_set_get_roundtrip(self):
        """``set_gates`` followed by ``get_gates`` should return the clipped values."""
        cost_fn = GOATCostFunction(num_heads=self.num_heads)
        new_gates = np.array([0.5, 1.5, 2.0, 3.0], dtype=np.float32)
        cost_fn.set_gates(new_gates)
        retrieved = cost_fn.get_gates()
        np.testing.assert_allclose(retrieved, new_gates, atol=1e-6)

    def test_compute_shape(self):
        """``compute`` should return a cost matrix with shape ``[B, heads, N, N]``."""
        cost_fn = GOATCostFunction(num_heads=self.num_heads)
        B, N, D = 2, 32, 32  # head_dim = D // num_heads = 8
        # GOATCostFunction.compute expects Q, K of shape [B, heads, N, head_dim]
        Q = np.random.randn(B, self.num_heads, N, D // self.num_heads).astype(
            np.float32
        )
        K = np.random.randn(B, self.num_heads, N, D // self.num_heads).astype(
            np.float32
        )
        C = cost_fn.compute(Q, K)
        self.assertEqual(C.shape, (B, self.num_heads, N, N))
        self.assertTrue(np.all(C >= 0.0), "Cost matrix should be non-negative")


if __name__ == "__main__":
    unittest.main(verbosity=2)
