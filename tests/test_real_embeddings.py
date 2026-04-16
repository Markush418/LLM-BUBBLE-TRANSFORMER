"""Test suite for real embedding edge cases and overflow handling.

This module tests the metrics pipeline against:
- Large embeddings (N > 1000)
- Float16 dtype (from quantized models)
- Extreme values (values > 1000)
- Mixed precision scenarios
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from metrics import (
    anisotropy_index,
    effective_rank,
    intrinsic_dimension_mle as intrinsic_dim_mle,
    concentration_ratio,
)


class TestLargeEmbeddings(unittest.TestCase):
    """Test metrics with large embeddings (real Qwen scenario)."""

    def test_anisotropy_index_large_float32(self):
        """Anisotropy index handles large embeddings in float32."""
        # Real Qwen: 1842 samples x 1024 dims
        large_emb = np.random.randn(1842, 1024).astype(np.float32)
        result = anisotropy_index(large_emb)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_anisotropy_index_large_float16(self):
        """Anisotropy index handles float16 embeddings (quantized models)."""
        # Float16 is common in quantized models
        large_emb = np.random.randn(1842, 1024).astype(np.float16)
        result = anisotropy_index(large_emb)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestExtremeValues(unittest.TestCase):
    """Test metrics with extreme values found in real embeddings."""

    def test_anisotropy_extreme_values_float32(self):
        """Anisotropy handles extreme values in float32."""
        # Real Qwen has values up to 6784
        large_emb = np.random.randn(1000, 512).astype(np.float32) * 100
        large_emb[0, 0] = 6784.0  # Extreme value
        result = anisotropy_index(large_emb)
        self.assertIsInstance(result, float)
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))

    def test_anisotropy_extreme_values_float16(self):
        """Anisotropy handles float16 with extreme values (up to max float16)."""
        # Float16 max is ~65504
        large_emb = np.random.randn(1000, 512).astype(np.float16) * 10
        large_emb[0, 0] = 600.0  # Within float16 range but large
        result = anisotropy_index(large_emb)
        self.assertIsInstance(result, float)
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))

    def test_effective_rank_extreme_values(self):
        """Effective rank handles embeddings with extreme values."""
        large_emb = np.random.randn(1000, 512).astype(np.float32) * 100
        result = effective_rank(large_emb)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_intrinsic_dim_extreme_values(self):
        """Intrinsic dimension handles embeddings with extreme values."""
        large_emb = np.random.randn(1000, 512).astype(np.float32) * 50
        result = intrinsic_dim_mle(large_emb)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)


class TestPrecisionHandling(unittest.TestCase):
    """Test that metrics gracefully handle precision conversions."""

    def test_float16_to_float64_conversion(self):
        """Metrics work correctly when float16 is promoted to float64."""
        # Float16 embeddings should be promoted internally
        emb_f16 = np.random.randn(500, 256).astype(np.float16)
        emb_f32 = emb_f16.astype(np.float32)

        result_f16 = anisotropy_index(emb_f16)
        result_f32 = anisotropy_index(emb_f32)

        # Results should be similar (not exact due to precision)
        self.assertAlmostEqual(result_f16, result_f32, places=2)

    def test_all_metrics_float16(self):
        """All metrics handle float16 input without overflow."""
        emb = np.random.randn(800, 512).astype(np.float16)

        # Should not raise
        result_aniso = anisotropy_index(emb)
        result_rank = effective_rank(emb)
        result_dim = intrinsic_dim_mle(emb)

        # Should be valid
        self.assertFalse(np.isnan(result_aniso))
        self.assertFalse(np.isnan(result_rank))
        self.assertFalse(np.isnan(result_dim))


class TestEdgeCases(unittest.TestCase):
    """Additional edge cases from real-world scenarios."""

    def test_very_small_embeddings(self):
        """Metrics handle very small values (near float16 epsilon)."""
        emb = np.random.randn(100, 128).astype(np.float16) * 0.001
        result = anisotropy_index(emb)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_nearly_constant_embeddings(self):
        """Metrics handle embeddings with near-zero variance."""
        emb = np.ones((200, 128), dtype=np.float32) * 0.5
        emb += np.random.randn(200, 128).astype(np.float32) * 0.001
        result = anisotropy_index(emb)
        # Should still produce valid result
        self.assertFalse(np.isnan(result))


if __name__ == "__main__":
    unittest.main()
