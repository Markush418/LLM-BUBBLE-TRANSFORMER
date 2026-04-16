"""
Test suite for Cost Functions — Plan E: Cost Matrix Engineering
=================================================================
15+ tests covering:
- All 5 cost functions: shape, non-negativity, convergence
- CostFunctionFactory
- Backward compatibility (L2SquaredCost vs _cdist_sq)
- Convergence validation
"""

import sys
import os
import unittest
import numpy as np

# Add experiments to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from plateau_attention import (
    CostFunction,
    CostFunctionFactory,
    VALID_COST_TYPES,
    L2SquaredCost,
    CosineCost,
    DotProductCost,
    MahalanobisCost,
    MeshLearnableCost,
    PlateauAttentionMechanism,
)


class TestCostFunctionFactory(unittest.TestCase):
    """Test the cost function factory."""

    def test_valid_cost_types_defined(self):
        """VALID_COST_TYPES contains exactly 5 cost types."""
        self.assertEqual(len(VALID_COST_TYPES), 5)
        self.assertIn("l2_sq", VALID_COST_TYPES)
        self.assertIn("cosine", VALID_COST_TYPES)
        self.assertIn("dot_product", VALID_COST_TYPES)
        self.assertIn("mahalanobis", VALID_COST_TYPES)
        self.assertIn("mesh_learnable", VALID_COST_TYPES)

    def test_create_l2_sq(self):
        """Factory creates L2SquaredCost."""
        cost_fn = CostFunctionFactory.create_cost("l2_sq")
        self.assertIsInstance(cost_fn, L2SquaredCost)

    def test_create_cosine(self):
        """Factory creates CosineCost."""
        cost_fn = CostFunctionFactory.create_cost("cosine")
        self.assertIsInstance(cost_fn, CosineCost)

    def test_create_dot_product(self):
        """Factory creates DotProductCost."""
        cost_fn = CostFunctionFactory.create_cost("dot_product")
        self.assertIsInstance(cost_fn, DotProductCost)

    def test_create_mahalanobis(self):
        """Factory creates MahalanobisCost."""
        cost_fn = CostFunctionFactory.create_cost("mahalanobis")
        self.assertIsInstance(cost_fn, MahalanobisCost)

    def test_create_mesh_learnable(self):
        """Factory creates MeshLearnableCost."""
        cost_fn = CostFunctionFactory.create_cost("mesh_learnable")
        self.assertIsInstance(cost_fn, MeshLearnableCost)

    def test_create_invalid_type_raises(self):
        """Factory raises ValueError for unknown cost type."""
        with self.assertRaises(ValueError):
            CostFunctionFactory.create_cost("invalid_type")


class TestL2SquaredCost(unittest.TestCase):
    """Test L2 squared distance cost function."""

    def setUp(self):
        self.cost_fn = L2SquaredCost()
        np.random.seed(42)
        self.B, self.H, self.N, self.D = 2, 4, 32, 32
        self.Q = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)
        self.K = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)

    def test_output_shape(self):
        """Output shape is [B, heads, N, N]."""
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertEqual(C.shape, (self.B, self.H, self.N, self.N))

    def test_output_dtype(self):
        """Output dtype is np.float32."""
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertEqual(C.dtype, np.float32)

    def test_non_negative(self):
        """All values are non-negative."""
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertTrue(np.all(C >= 0))

    def test_no_nan_inf(self):
        """No NaN or Inf values."""
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertFalse(np.any(np.isnan(C)))
        self.assertFalse(np.any(np.isinf(C)))

    def test_backward_compat_with_cdist_sq(self):
        """L2SquaredCost output is byte-identical to PlateauAttentionMechanism._cdist_sq."""
        attn = PlateauAttentionMechanism(d_model=128, num_heads=4)
        C_new = self.cost_fn.compute(self.Q, self.K)
        C_orig = attn._cdist_sq(self.Q, self.K)
        max_diff = np.max(np.abs(C_new - C_orig))
        self.assertLess(max_diff, 1e-8, f"Max diff: {max_diff}")


class TestCosineCost(unittest.TestCase):
    """Test cosine distance cost function."""

    def setUp(self):
        self.cost_fn = CosineCost()
        np.random.seed(42)
        self.B, self.H, self.N, self.D = 2, 4, 32, 32
        self.Q = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)
        self.K = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)

    def test_output_shape(self):
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertEqual(C.shape, (self.B, self.H, self.N, self.N))

    def test_non_negative(self):
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertTrue(np.all(C >= 0))

    def test_bounded_range(self):
        """Cosine distance is in [0, 2]."""
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertTrue(np.all(C >= 0))
        self.assertTrue(np.all(C <= 2.0 + 1e-6))

    def test_identical_vectors_zero_cost(self):
        """Identical Q and K should produce near-zero cost on diagonal."""
        Q = np.random.randn(1, 1, 5, 8).astype(np.float32)
        C = self.cost_fn.compute(Q, Q)
        # Diagonal should be near zero (cos(Q_i, Q_i) = 1, so 1-1 = 0)
        diag = np.diagonal(C[0, 0], axis1=-2, axis2=-1)
        self.assertTrue(np.all(diag < 1e-6))


class TestDotProductCost(unittest.TestCase):
    """Test negative dot product cost function."""

    def setUp(self):
        self.cost_fn = DotProductCost()
        np.random.seed(42)
        self.B, self.H, self.N, self.D = 2, 4, 32, 32
        self.Q = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)
        self.K = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)

    def test_output_shape(self):
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertEqual(C.shape, (self.B, self.H, self.N, self.N))

    def test_non_negative(self):
        """Shifted dot product must be non-negative."""
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertTrue(np.all(C >= 0))

    def test_no_nan_inf(self):
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertFalse(np.any(np.isnan(C)))
        self.assertFalse(np.any(np.isinf(C)))


class TestMahalanobisCost(unittest.TestCase):
    """Test Mahalanobis distance cost function."""

    def setUp(self):
        self.cost_fn = MahalanobisCost(reg_lambda=1e-6)
        np.random.seed(42)
        self.B, self.H, self.N, self.D = 2, 4, 32, 32
        self.Q = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)
        self.K = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)

    def test_output_shape(self):
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertEqual(C.shape, (self.B, self.H, self.N, self.N))

    def test_non_negative(self):
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertTrue(np.all(C >= 0))

    def test_no_nan_inf(self):
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertFalse(np.any(np.isnan(C)))
        self.assertFalse(np.any(np.isinf(C)))


class TestMeshLearnableCost(unittest.TestCase):
    """Test MESH-style learnable cost function."""

    def setUp(self):
        self.cost_fn = MeshLearnableCost(alpha=0.1, seed=42)
        np.random.seed(42)
        self.B, self.H, self.N, self.D = 2, 4, 32, 32
        self.Q = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)
        self.K = np.random.randn(self.B, self.H, self.N, self.D).astype(np.float32)

    def test_output_shape(self):
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertEqual(C.shape, (self.B, self.H, self.N, self.N))

    def test_non_negative(self):
        C = self.cost_fn.compute(self.Q, self.K)
        self.assertTrue(np.all(C >= 0))

    def test_delta_initialized(self):
        """Delta should be initialized after first compute."""
        self.cost_fn.compute(self.Q, self.K)
        self.assertIsNotNone(self.cost_fn.delta)


class TestConvergenceValidation(unittest.TestCase):
    """Test Sinkhorn convergence validation."""

    def test_perfect_doubly_stochastic(self):
        """Identity-like matrix should pass convergence."""
        attn = PlateauAttentionMechanism(d_model=128, num_heads=4, epsilon=0.5)
        np.random.seed(42)
        x = np.random.randn(2, 32, 128).astype(np.float32)
        _, A = attn.forward(x, return_attention=True)
        is_valid, max_dev = attn._validate_convergence(A)
        self.assertTrue(is_valid, f"Max deviation: {max_dev}")

    def test_uniform_matrix_passes(self):
        """Uniform matrix (all 1/N) should pass convergence for N=32."""
        attn = PlateauAttentionMechanism(d_model=128, num_heads=4)
        A = np.ones((2, 4, 32, 32), dtype=np.float32) / 32.0
        is_valid, max_dev = attn._validate_convergence(A)
        # Uniform matrix has row/col sums = 1.0, so it should pass
        self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
