"""Tests for SIRI post-processing module."""

import sys
import os
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from siri_postprocess import (
    siri_sinkhorn_log_domain,
    siri_postprocess_attention,
    siri_interpolate,
)


class TestSinkhornLogDomain(unittest.TestCase):
    """Tests for the log-domain Sinkhorn-Knopp implementation."""

    def setUp(self):
        np.random.seed(42)
        self.B, self.H, self.N = 2, 4, 16
        self.head_dim = 32
        Q = np.random.randn(self.B, self.H, self.N, self.head_dim).astype(np.float32) * 0.1
        K = np.random.randn(self.B, self.H, self.N, self.head_dim).astype(np.float32) * 0.1
        # log_S = -||Q-K||^2 / eps
        diff = Q[:, :, :, np.newaxis, :] - K[:, :, np.newaxis, :, :]
        C = np.sum(diff ** 2, axis=-1)
        self.log_S = -C / 0.1

    def test_doubly_stochastic(self):
        """After Sinkhorn iterations, A should be approximately doubly stochastic."""
        A = siri_sinkhorn_log_domain(self.log_S, tau_iters=10)
        row_sums = A.sum(axis=-1)
        col_sums = A.sum(axis=-2)
        np.testing.assert_allclose(row_sums, 1.0, atol=5e-2)
        np.testing.assert_allclose(col_sums, 1.0, atol=5e-2)

    def test_output_shape(self):
        """Output A should have shape [B, H, N, N]."""
        A = siri_sinkhorn_log_domain(self.log_S, tau_iters=5)
        self.assertEqual(A.shape, (self.B, self.H, self.N, self.N))

    def test_non_negative(self):
        """All values should be non-negative (probabilities)."""
        A = siri_sinkhorn_log_domain(self.log_S, tau_iters=5)
        self.assertGreaterEqual(A.min(), 0.0)

    def test_with_mask(self):
        """Mask should zero out masked positions."""
        mask = np.ones((self.N, self.N), dtype=np.float32)
        # Mask only a few specific positions (not entire row, which would break Sinkhorn).
        mask[0, 0] = 0
        mask[1, 3] = 0
        mask[5, 5] = 0
        A = siri_sinkhorn_log_domain(self.log_S, tau_iters=10, mask=mask)
        # Masked positions should be near zero
        self.assertLess(A[0, 0, 0, 0], 1e-3)
        self.assertLess(A[0, 0, 1, 3], 1e-3)
        self.assertLess(A[0, 0, 5, 5], 1e-3)
        # Other positions should be nonzero
        self.assertGreater(A[0, 0, 0, 1], 1e-3)


class TestSIRIPostprocess(unittest.TestCase):
    """Tests for siri_postprocess_attention."""

    def setUp(self):
        np.random.seed(42)
        self.B, self.H, self.N = 2, 4, 16
        self.head_dim = 32
        self.Q = np.random.randn(self.B, self.H, self.N, self.head_dim).astype(np.float32) * 0.1
        self.K = np.random.randn(self.B, self.H, self.N, self.head_dim).astype(np.float32) * 0.1
        self.V = np.random.randn(self.B, self.H, self.N, self.head_dim).astype(np.float32) * 0.1
        # SOTA attention: row-stochastic softmax
        scores = np.random.randn(self.B, self.H, self.N, self.N).astype(np.float32) * 0.5
        self.A_sota = np.exp(scores - scores.max(axis=-1, keepdims=True))
        self.A_sota = self.A_sota / self.A_sota.sum(axis=-1, keepdims=True)

    def test_output_shape(self):
        """A_post should have shape [B, H, N, N]."""
        A_post = siri_postprocess_attention(self.A_sota, self.Q, self.K, epsilon=0.1)
        self.assertEqual(A_post.shape, (self.B, self.H, self.N, self.N))

    def test_row_stochastic(self):
        """A_post should be row-stochastic (each row sums to ~1)."""
        A_post = siri_postprocess_attention(self.A_sota, self.Q, self.K, epsilon=0.1)
        row_sums = A_post.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_dense_then_sparse(self):
        """If A_sota is dense, A_post should still be dense (but refined)."""
        A_post = siri_postprocess_attention(self.A_sota, self.Q, self.K, epsilon=0.1)
        self.assertGreater(A_post.mean(), 0.0)

    def test_sparse_then_sparser(self):
        """If A_sota is sparse (one-hot), A_post should also be sparse."""
        # One-hot attention
        A_onehot = np.zeros((self.B, self.H, self.N, self.N), dtype=np.float32)
        A_onehot[:, :, :, 0] = 1.0
        A_post = siri_postprocess_attention(A_onehot, self.Q, self.K, epsilon=0.1)
        # Should still have most mass in first column
        self.assertGreater(A_post[:, :, :, 0].sum(), 0.5 * self.B * self.H * self.N)


class TestSIRIInterpolate(unittest.TestCase):
    """Tests for siri_interpolate."""

    def setUp(self):
        np.random.seed(42)
        self.B, self.N, self.d = 2, 16, 64

    def test_lam_one_returns_sota(self):
        """lam=1.0 should return exactly out_sota."""
        out_sota = np.random.randn(self.B, self.N, self.d).astype(np.float32)
        out_siri = np.random.randn(self.B, self.N, self.d).astype(np.float32)
        out = siri_interpolate(out_sota, out_siri, lam=1.0)
        np.testing.assert_array_equal(out, out_sota)

    def test_lam_zero_returns_siri(self):
        """lam=0.0 should return exactly out_siri."""
        out_sota = np.random.randn(self.B, self.N, self.d).astype(np.float32)
        out_siri = np.random.randn(self.B, self.N, self.d).astype(np.float32)
        out = siri_interpolate(out_sota, out_siri, lam=0.0)
        np.testing.assert_array_equal(out, out_siri)

    def test_lam_half_returns_average(self):
        """lam=0.5 should return average."""
        out_sota = np.random.randn(self.B, self.N, self.d).astype(np.float32)
        out_siri = np.random.randn(self.B, self.N, self.d).astype(np.float32)
        out = siri_interpolate(out_sota, out_siri, lam=0.5)
        expected = 0.5 * out_sota + 0.5 * out_siri
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_invalid_lam_raises(self):
        """lam outside [0, 1] should raise ValueError."""
        out_sota = np.random.randn(self.B, self.N, self.d).astype(np.float32)
        out_siri = np.random.randn(self.B, self.N, self.d).astype(np.float32)
        with self.assertRaises(ValueError):
            siri_interpolate(out_sota, out_siri, lam=1.5)
        with self.assertRaises(ValueError):
            siri_interpolate(out_sota, out_siri, lam=-0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
