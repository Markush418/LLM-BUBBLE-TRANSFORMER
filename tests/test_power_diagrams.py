"""Tests for PowerDiagramModule (psi as Laguerre tessellation bias)."""

import sys
import os
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from power_diagrams import (
    PowerDiagramModule,
    compute_psi_from_keys,
    apply_psi_to_log_sinkhorn,
    power_diagram_assign,
)


class TestComputePsi(unittest.TestCase):
    """Tests for psi computation."""

    def setUp(self):
        self.B, self.N, self.d_model = 2, 16, 64

    def test_compute_psi_shape(self):
        """psi should have shape [B, N, 1]."""
        K = torch.randn(self.B, self.N, self.d_model)
        W_psi = np.random.randn(self.d_model, 1).astype(np.float32) * 0.1
        psi = compute_psi_from_keys(K, W_psi)
        self.assertEqual(psi.shape, (self.B, self.N, 1))

    def test_compute_psi_from_torch(self):
        """torch tensor K should be converted internally."""
        K = torch.randn(self.B, self.N, self.d_model)
        W_psi = np.random.randn(self.d_model, 1).astype(np.float32) * 0.1
        psi = compute_psi_from_keys(K, W_psi)
        self.assertIsInstance(psi, np.ndarray)
        self.assertEqual(psi.shape, (self.B, self.N, 1))


class TestApplyPsi(unittest.TestCase):
    """Tests for applying psi bias to log_Sinkhorn."""

    def setUp(self):
        self.B, self.heads, self.N = 2, 4, 16
        self.log_S = torch.randn(self.B, self.heads, self.N, self.N)

    def test_apply_psi_shape_preserved(self):
        """log_S_psi should have same shape as log_S."""
        psi = torch.randn(self.B, self.N, 1)
        log_S_psi = apply_psi_to_log_sinkhorn(self.log_S, psi)
        self.assertEqual(log_S_psi.shape, self.log_S.shape)

    def test_apply_psi_zero_preserves(self):
        """psi=0 should leave log_S unchanged."""
        psi = torch.zeros(self.B, self.N, 1)
        log_S_psi = apply_psi_to_log_sinkhorn(self.log_S, psi)
        np.testing.assert_allclose(log_S_psi, self.log_S.numpy(), atol=1e-6)

    def test_apply_psi_adds_correctly(self):
        """psi should be added to the column dimension (key axis)."""
        psi = torch.randn(self.B, self.N, 1)
        log_S_psi = apply_psi_to_log_sinkhorn(self.log_S, psi)
        # Check that log_S_psi[b, h, i, j] == log_S[b, h, i, j] + psi[b, j, 0]
        expected = self.log_S.numpy() + psi.numpy()[:, np.newaxis, :, :]
        np.testing.assert_allclose(log_S_psi, expected, atol=1e-6)


class TestPowerDiagramAssignment(unittest.TestCase):
    """Tests for Power Diagram assignment."""

    def test_voronoi_special_case(self):
        """psi=0 should reduce to Voronoi assignment."""
        np.random.seed(42)
        B, N, d = 1, 20, 8
        K = np.random.randn(B, N, d)
        centroids = np.random.randn(5, d)
        psi = np.zeros(5)

        assignments = power_diagram_assign(K, centroids, psi)
        self.assertEqual(assignments.shape, (B, N))

        # Each point assigned to one of K cells
        for b in range(B):
            self.assertTrue(np.all(assignments[b] >= 0))
            self.assertTrue(np.all(assignments[b] < 5))

    def test_psi_changes_assignment(self):
        """Different psi should produce different assignments."""
        np.random.seed(42)
        B, N, d = 1, 20, 8
        K = np.random.randn(B, N, d)
        centroids = np.random.randn(5, d)
        psi_a = np.zeros(5)
        psi_b = np.random.randn(5) * 2.0

        assignments_a = power_diagram_assign(K, centroids, psi_a)
        assignments_b = power_diagram_assign(K, centroids, psi_b)
        # At least some assignments should differ
        self.assertFalse(np.array_equal(assignments_a, assignments_b))


class TestPowerDiagramModule(unittest.TestCase):
    """Tests for the learnable PowerDiagramModule."""

    def test_module_initialization(self):
        """Module should initialize W_psi with correct shape."""
        pd = PowerDiagramModule(d_model=128, seed=42)
        self.assertEqual(pd.W_psi.shape, (128, 1))

    def test_compute_psi(self):
        """compute_psi should return [B, N, 1]."""
        pd = PowerDiagramModule(d_model=64, seed=42)
        K = torch.randn(2, 16, 64)
        psi = pd.compute_psi(K)
        self.assertEqual(psi.shape, (2, 16, 1))

    def test_apply_to_log_sinkhorn(self):
        """apply_to_log_sinkhorn should add bias correctly."""
        pd = PowerDiagramModule(d_model=64, seed=42)
        log_S = torch.randn(2, 4, 16, 16)
        K = torch.randn(2, 16, 64)
        log_S_psi = pd.apply_to_log_sinkhorn(log_S, K)
        self.assertEqual(log_S_psi.shape, log_S.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
