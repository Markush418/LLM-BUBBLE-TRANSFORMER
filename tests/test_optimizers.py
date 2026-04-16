"""
Tests for Riemannian Optimizer Utilities
=========================================

Unit tests for:
- get_manifold_parameters: Extract ManifoldParameters from module
- get_regular_parameters: Extract regular parameters from module
- create_riemannian_optimizer: Create RiemannianAdam/Adam optimizer
- create_optimizer_with_separate_groups: Optimizer with separate LR groups
"""

import sys
import os
import unittest
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.optimizers import (
    get_manifold_parameters,
    get_regular_parameters,
    create_riemannian_optimizer,
    create_optimizer_with_separate_groups,
    GEOOPT_AVAILABLE,
)
from models.bubble_centroids_v4 import (
    BubbleCentroidsV4,
    ManifoldType,
)


class TestRiemannianOptimizer(unittest.TestCase):
    """Tests for Riemannian optimizer utilities."""

    def test_get_manifold_parameters_detects_manifold(self):
        """Should detect ManifoldParameters in BubbleCentroidsV4."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        # Create module with Poincaré manifold
        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.POINCARE,
        )

        manifold_params = get_manifold_parameters(module)

        # Should detect centroids as ManifoldParameter
        self.assertGreater(len(manifold_params), 0)
        self.assertTrue(all(hasattr(p, "manifold") for p in manifold_params))

    def test_get_regular_parameters_excludes_manifold(self):
        """Should separate regular from manifold params."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        # Create module with Poincaré manifold
        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.POINCARE,
        )

        manifold_params = get_manifold_parameters(module)
        regular_params = get_regular_parameters(module)

        # Regular params should not include manifold params
        manifold_ids = set(id(p) for p in manifold_params)
        regular_ids = set(id(p) for p in regular_params)

        # No overlap
        self.assertEqual(len(manifold_ids & regular_ids), 0)

        # All params accounted for
        all_params = list(module.parameters())
        self.assertEqual(len(manifold_params) + len(regular_params), len(all_params))

    def test_create_riemannian_optimizer_creation(self):
        """Should create optimizer for manifold module."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.POINCARE,
        )

        optimizer = create_riemannian_optimizer(module, lr=1e-3)

        # Should be RiemannianAdam (geoopt optimizer)
        self.assertIsNotNone(optimizer)
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_create_riemannian_optimizer_fallback(self):
        """Should fall back to Adam when geoopt unavailable."""
        # Create Euclidean module (no manifold)
        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.EUCLIDEAN,
        )

        optimizer = create_riemannian_optimizer(module, lr=1e-3)

        # Should always be valid optimizer
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, torch.optim.Optimizer)

        # If geoopt not available, should be Adam
        if not GEOOPT_AVAILABLE:
            self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_create_riemannian_optimizer_no_manifold_params(self):
        """Should return Adam for regular module."""

        # Simple module with no ManifoldParameters
        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(64, 64))

        module = SimpleModule()
        optimizer = create_riemannian_optimizer(module, lr=1e-3)

        # Should be Adam (no manifold params)
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_optimizer_step_preserves_manifold(self):
        """Optimizer step should keep centroids on manifold."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.POINCARE,
        )

        optimizer = create_riemannian_optimizer(module, lr=1e-3)

        # Get initial centroids
        centroids_before = module.centroids.clone()

        # Simulate training step
        loss = module.centroids.sum()
        loss.backward()
        optimizer.step()

        # Centroids should still be on manifold
        # For Poincaré ball, norm should be < 1
        norms = torch.norm(module.centroids, dim=-1)
        self.assertTrue((norms < 1.0).all())

        # Centroids should have changed
        self.assertFalse(torch.allclose(centroids_before, module.centroids))

    def test_separate_learning_rates(self):
        """create_optimizer_with_separate_groups should work."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.POINCARE,
        )

        optimizer = create_optimizer_with_separate_groups(
            module,
            lr_manifold=1e-3,
            lr_regular=1e-4,
        )

        # Should have parameter groups
        self.assertGreater(len(optimizer.param_groups), 0)

        # Check learning rates are set
        for group in optimizer.param_groups:
            self.assertIn("lr", group)

    def test_optimizer_with_sgd(self):
        """Should support SGD optimizer type."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.POINCARE,
        )

        optimizer = create_riemannian_optimizer(module, lr=1e-3, optimizer_type="sgd")

        # Should be RiemannianSGD (geoopt optimizer)
        self.assertIsNotNone(optimizer)
        self.assertEqual(len(optimizer.param_groups), 1)


class TestOptimizerEdgeCases(unittest.TestCase):
    """Tests for edge cases in optimizer creation."""

    def test_empty_module(self):
        """Should raise ValueError for module with no parameters."""

        class EmptyModule(nn.Module):
            pass

        module = EmptyModule()

        # PyTorch raises ValueError for empty parameter list
        with self.assertRaises(ValueError):
            create_riemannian_optimizer(module, lr=1e-3)

    def test_invalid_optimizer_type(self):
        """Should raise ValueError for unknown optimizer type."""
        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.EUCLIDEAN,
        )

        with self.assertRaises(ValueError):
            create_riemannian_optimizer(module, lr=1e-3, optimizer_type="invalid")

    def test_weight_decay_parameter(self):
        """Should pass weight_decay to optimizer."""
        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.EUCLIDEAN,
        )

        weight_decay = 0.01
        optimizer = create_riemannian_optimizer(
            module, lr=1e-3, weight_decay=weight_decay
        )

        # Check weight_decay is set
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], weight_decay)

    def test_separate_weight_decay(self):
        """Should support separate weight decay for manifold/regular params."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.POINCARE,
        )

        optimizer = create_optimizer_with_separate_groups(
            module,
            lr_manifold=1e-3,
            lr_regular=1e-4,
            weight_decay_manifold=0.01,
            weight_decay_regular=0.001,
        )

        # Should have parameter groups with different weight decay
        self.assertGreater(len(optimizer.param_groups), 0)


class TestManifoldParameterDetection(unittest.TestCase):
    """Tests for ManifoldParameter detection logic."""

    def test_detect_sphere_manifold(self):
        """Should detect Sphere manifold parameters."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.SPHERE,
        )

        manifold_params = get_manifold_parameters(module)

        # Should detect centroids as ManifoldParameter
        self.assertGreater(len(manifold_params), 0)

    def test_euclidean_no_manifold_params(self):
        """Euclidean module should have no ManifoldParameters."""
        module = BubbleCentroidsV4(
            num_heads=8,
            num_experts=32,
            head_dim=64,
            manifold_type=ManifoldType.EUCLIDEAN,
        )

        manifold_params = get_manifold_parameters(module)

        # Euclidean should have no manifold params
        self.assertEqual(len(manifold_params), 0)

    def test_mixed_module_detection(self):
        """Should correctly detect manifold params in mixed module."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        class MixedModule(nn.Module):
            def __init__(self):
                super().__init__()
                # Regular parameter
                self.regular_weight = nn.Parameter(torch.randn(64, 64))
                # Manifold parameter (Poincaré)
                self.poincare_centroids = BubbleCentroidsV4(
                    num_heads=4,
                    num_experts=16,
                    head_dim=64,
                    manifold_type=ManifoldType.POINCARE,
                )

        module = MixedModule()
        manifold_params = get_manifold_parameters(module)

        # Should detect manifold params from nested module
        self.assertGreater(len(manifold_params), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
