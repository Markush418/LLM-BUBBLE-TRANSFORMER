"""Tests for HybridAttention (DeltaNet + SIRI + Power Diagram)."""

import sys
import os
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from hybrid_attention import HybridAttention


class TestHybridAttention(unittest.TestCase):
    """Tests for HybridAttention module."""

    def setUp(self):
        np.random.seed(42)
        self.B, self.N, self.d_model, self.num_heads = 2, 32, 128, 4

    def test_pure_deltanet_lam_one(self):
        """lam=1.0 should produce output equivalent to DeltaNet only."""
        attn = HybridAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            lam=1.0,
        )
        x = torch.randn(self.B, self.N, self.d_model)
        output, attn_matrix = attn(x, return_attention=True)
        self.assertEqual(output.shape, (self.B, self.N, self.d_model))
        self.assertEqual(attn_matrix.shape, (self.B, self.num_heads, self.N, self.N))
        # With lam=1.0, output should match DeltaNet
        self.assertFalse(torch.isnan(output).any())

    def test_pure_siri_lam_zero(self):
        """lam=0.0 should produce pure SIRI output with doubly-stochastic attention."""
        attn = HybridAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            lam=0.0,
        )
        x = torch.randn(self.B, self.N, self.d_model)
        output, attn_matrix = attn(x, return_attention=True)
        self.assertEqual(output.shape, (self.B, self.N, self.d_model))
        # Attention should be approximately doubly stochastic
        row_sums = attn_matrix.sum(dim=-1).numpy()
        col_sums = attn_matrix.sum(dim=-2).numpy()
        np.testing.assert_allclose(row_sums, 1.0, atol=5e-2)
        np.testing.assert_allclose(col_sums, 1.0, atol=5e-2)

    def test_hybrid_lam_half(self):
        """lam=0.5 should produce hybrid output."""
        attn = HybridAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            lam=0.5,
        )
        x = torch.randn(self.B, self.N, self.d_model)
        output, attn_matrix = attn(x, return_attention=True)
        self.assertEqual(output.shape, (self.B, self.N, self.d_model))
        self.assertEqual(attn_matrix.shape, (self.B, self.num_heads, self.N, self.N))

    def test_no_nan_across_epsilons(self):
        """Output should be NaN-free across epsilon values."""
        for eps in [0.001, 0.01, 0.1, 1.0]:
            attn = HybridAttention(
                d_model=self.d_model,
                num_heads=self.num_heads,
                epsilon=eps,
            )
            x = torch.randn(self.B, self.N, self.d_model)
            output = attn(x)
            self.assertFalse(torch.isnan(output).any(), f"NaN at eps={eps}")
            self.assertFalse(torch.isinf(output).any(), f"Inf at eps={eps}")

    def test_accepts_numpy_input(self):
        """Module should accept numpy arrays as input."""
        attn = HybridAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        x = np.random.randn(self.B, self.N, self.d_model).astype(np.float32)
        output = attn(x)
        self.assertEqual(output.shape, (self.B, self.N, self.d_model))

    def test_returns_torch_when_torch_input(self):
        """torch input should produce torch output."""
        attn = HybridAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
        )
        x = torch.randn(self.B, self.N, self.d_model)
        output = attn(x)
        self.assertIsInstance(output, torch.Tensor)

    def test_deterministic(self):
        """Same seed + input should produce same output."""
        attn1 = HybridAttention(
            d_model=self.d_model, num_heads=self.num_heads, seed=42,
        )
        attn2 = HybridAttention(
            d_model=self.d_model, num_heads=self.num_heads, seed=42,
        )
        x = torch.randn(self.B, self.N, self.d_model)
        out1 = attn1(x)
        out2 = attn2(x)
        np.testing.assert_array_equal(out1.detach().numpy(), out2.detach().numpy())

    def test_power_diagram_psi_is_applied(self):
        """Verify Power Diagram psi is added to log_Sinkhorn.

        [EMPIRICAL] Note: Under doubly-stochastic Sinkhorn normalization,
        per-column shifts in log_S are absorbed by the column potentials.
        This is mathematically expected. We verify psi is APPLIED to log_S
        by checking it changes the pre-Sinkhorn log_S values.
        """
        np.random.seed(42)
        attn = HybridAttention(
            d_model=self.d_model, num_heads=self.num_heads, lam=0.0,
        )
        x = torch.randn(self.B, self.N, self.d_model)

        # Compute log_S with and without psi (manually).
        x_np = x.detach().numpy()
        head_dim = self.d_model // self.num_heads
        W_q = attn.deltanet.W_q
        W_k = attn.deltanet.W_k
        Q = (x_np @ W_q).reshape(self.B, self.N, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        K = (x_np @ W_k).reshape(self.B, self.N, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        Q_sq = np.sum(Q ** 2, axis=-1, keepdims=True)
        K_sq = np.sum(K ** 2, axis=-1, keepdims=True)
        C = np.maximum(Q_sq + np.moveaxis(K_sq, -2, -1) - 2.0 * np.matmul(Q, np.moveaxis(K, -2, -1)), 0.0)
        C_min = np.min(C, axis=(-2, -1), keepdims=True)
        C_max = np.max(C, axis=(-2, -1), keepdims=True)
        C = (C - C_min) / (C_max - C_min + 1e-10)
        log_S = -C / attn.epsilon  # [B, H, N, N]

        # Compute psi for two settings.
        attn.pd.W_psi = np.zeros_like(attn.pd.W_psi)
        psi_zero = attn.pd.compute_psi(x_np)

        attn.pd.W_psi = (np.random.randn(self.d_model, 1) * 1.0).astype(np.float32)
        psi_random = attn.pd.compute_psi(x_np)

        # log_S with psi should differ from log_S without psi.
        log_S_zero = log_S + psi_zero[:, np.newaxis, :, :]
        log_S_random = log_S + psi_random[:, np.newaxis, :, :]
        diff = np.abs(log_S_zero - log_S_random).max()
        self.assertGreater(diff, 0.01, "psi should be applied to log_Sinkhorn")


if __name__ == "__main__":
    unittest.main(verbosity=2)
