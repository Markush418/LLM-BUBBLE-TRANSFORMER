"""Tests for SIRI-Soft variants (siri_soft.py).

Conventions: unittest framework, sibling imports.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from siri_soft import siri_soft_blend, siri_sparse, siri_chiller


def make_scores(B=2, H=4, N=16, peak_pos=(0, 0, 0), peak_value=8.0, seed=42):
    """Generate 4D attention scores with a single dominant peak."""
    rng = np.random.RandomState(seed)
    scores = rng.randn(B, H, N, N).astype(np.float32)
    scores[peak_pos] += peak_value
    return scores


def make_causal_mask(N=16):
    """Causal mask [N, N] additive (0 valid, -inf masked)."""
    mask = np.triu(np.full((N, N), float("-inf"), dtype=np.float32), k=1)
    return mask


def row_entropy(A):
    """Average per-row entropy (max = log(N))."""
    A_safe = np.maximum(A, 1e-12)
    return -(A_safe * np.log(A_safe)).sum(axis=-1).mean()


class TestSiriSoftShapes(unittest.TestCase):
    def setUp(self):
        self.scores = make_scores()

    def test_soft_blend_shape(self):
        A = siri_soft_blend(self.scores, alpha=0.3)
        self.assertEqual(A.shape, self.scores.shape)

    def test_sparse_shape(self):
        A = siri_sparse(self.scores)
        self.assertEqual(A.shape, self.scores.shape)

    def test_chiller_shape(self):
        A = siri_chiller(self.scores, beta=10.0)
        self.assertEqual(A.shape, self.scores.shape)


class TestRowStochastic(unittest.TestCase):
    def setUp(self):
        self.scores = make_scores()

    def test_soft_blend_row_sums_to_one(self):
        A = siri_soft_blend(self.scores, alpha=0.5)
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, rtol=1e-5)

    def test_sparse_row_sums_to_one(self):
        A = siri_sparse(self.scores, tau_iters=20)
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, rtol=1e-5)

    def test_chiller_row_sums_to_one(self):
        A = siri_chiller(self.scores, beta=10.0)
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, rtol=1e-5)

    def test_chiller_high_beta_still_sums_to_one(self):
        """Even at high beta, row sums must = 1.0 (we renormalize)."""
        A = siri_chiller(self.scores, beta=50.0, tau_iters=20)
        np.testing.assert_allclose(A.sum(axis=-1), 1.0, rtol=1e-4)


class TestPeakedness(unittest.TestCase):
    """Chiller/Sparse should be more peaked than softmax."""

    def setUp(self):
        self.scores = make_scores()

    def _softmax(self, x):
        x_max = x.max(axis=-1, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=-1, keepdims=True)

    def test_chiller_more_peaked_than_softmax(self):
        A_sm = self._softmax(self.scores)
        A_chill = siri_chiller(self.scores, beta=5.0)
        self.assertLess(row_entropy(A_chill), row_entropy(A_sm))

    def test_sparse_more_peaked_than_softmax(self):
        A_sm = self._softmax(self.scores)
        A_sparse = siri_sparse(self.scores, tau_iters=20)
        self.assertLess(row_entropy(A_sparse), row_entropy(A_sm))

    def test_chiller_beta_sweep_increases_peakedness(self):
        A_low = siri_chiller(self.scores, beta=2.0)
        A_high = siri_chiller(self.scores, beta=20.0)
        self.assertLess(row_entropy(A_high), row_entropy(A_low))


class TestSoftBlendBehavior(unittest.TestCase):
    def setUp(self):
        self.scores = make_scores()

    def _softmax(self, x):
        x_max = x.max(axis=-1, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=-1, keepdims=True)

    def test_alpha_zero_equals_softmax(self):
        """alpha=0 should give standard softmax (within Sinkhorn roundoff)."""
        A_blend = siri_soft_blend(self.scores, alpha=0.0, tau_iters=5)
        A_sm = self._softmax(self.scores)
        np.testing.assert_allclose(A_blend, A_sm, rtol=1e-4)


class TestCausalMask(unittest.TestCase):
    def setUp(self):
        self.scores = make_scores()
        self.mask = make_causal_mask()

    def _check_zero_above_diagonal(self, A):
        N = A.shape[-1]
        for i in range(1, N):
            for j in range(i + 1, N):
                self.assertLess(A[..., i, j].max(), 1e-4)

    def test_soft_blend_respects_mask(self):
        A = siri_soft_blend(self.scores, alpha=0.5, mask=self.mask, tau_iters=10)
        self._check_zero_above_diagonal(A)

    def test_sparse_respects_mask(self):
        A = siri_sparse(self.scores, mask=self.mask, tau_iters=20)
        self._check_zero_above_diagonal(A)

    def test_chiller_respects_mask(self):
        A = siri_chiller(self.scores, beta=5.0, mask=self.mask, tau_iters=10)
        self._check_zero_above_diagonal(A)


class TestNumericalStability(unittest.TestCase):
    def setUp(self):
        self.scores = make_scores()

    def test_chiller_no_nan_high_beta(self):
        """High beta should not produce NaN/Inf."""
        A = siri_chiller(self.scores, beta=50.0, tau_iters=10)
        self.assertTrue(np.isfinite(A).all())

    def test_sparse_no_nan(self):
        A = siri_sparse(self.scores, tau_iters=20)
        self.assertTrue(np.isfinite(A).all())

    def test_soft_blend_no_nan_with_mask(self):
        A = siri_soft_blend(self.scores, alpha=0.5, mask=make_causal_mask(), tau_iters=10)
        self.assertTrue(np.isfinite(A).all())


class TestTorchCompatibility(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            self.torch = None

    def test_accepts_torch_tensor_chiller(self):
        if self.torch is None:
            self.skipTest("torch not available")
        scores_np = make_scores()
        scores_torch = self.torch.from_numpy(scores_np)
        A = siri_chiller(scores_torch, beta=5.0)
        self.assertEqual(A.shape, scores_np.shape)
        self.assertTrue(np.isfinite(A).all())

    def test_accepts_torch_tensor_sparse(self):
        if self.torch is None:
            self.skipTest("torch not available")
        scores_np = make_scores()
        scores_torch = self.torch.from_numpy(scores_np)
        A = siri_sparse(scores_torch, tau_iters=10)
        self.assertEqual(A.shape, scores_np.shape)


if __name__ == "__main__":
    unittest.main()