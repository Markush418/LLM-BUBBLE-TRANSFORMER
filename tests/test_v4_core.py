"""
Tests for Bubble Transformer V4 Core — FPS + Expert-Choice Routing
=================================================================

Unit tests for:
- fps_sample: Farthest Point Sampling
- fps_initialize_centroids: FPS-based centroid initialization
- expert_choice_routing: Expert-Choice routing mechanism
- routed_attention: Attention with routing masks
- FPSExpertChoiceAttention: Complete V4 module
- BubbleCentroidsV4: Manifold-aware centroids
"""

import sys
import os
import unittest
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.v4_core import (
    fps_sample,
    fps_initialize_centroids,
    expert_choice_routing,
    routed_attention,
    compute_routing_balance,
    compute_coverage,
    FPSExpertChoiceAttention,
    power_diagram_assign,
    warm_start_centroids,
    soft_sort,
)
from models.bubble_centroids_v4 import (
    BubbleCentroidsV4,
    HybridManifoldCentroids,
    ManifoldType,
    manifold_distance,
    project_to_manifold,
    get_manifold,
    GEOOPT_AVAILABLE,
)


class TestFPSSample(unittest.TestCase):
    """Tests for fps_sample function."""

    def test_fps_sample_shape(self):
        """FPS must produce C indices."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        points = torch.randn(B, H, N, d)
        indices = fps_sample(points, C)

        self.assertEqual(indices.shape, (B, H, C))

    def test_fps_sample_valid_indices(self):
        """All indices must be valid (in range [0, N))."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        points = torch.randn(B, H, N, d)
        indices = fps_sample(points, C)

        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < N).all())

    def test_fps_sample_unique_indices(self):
        """FPS should select unique points (no duplicates)."""
        B, H, N, d = 1, 1, 100, 64
        C = 32

        points = torch.randn(B, H, N, d)
        indices = fps_sample(points, C)

        # Check uniqueness
        unique_indices = torch.unique(indices)
        self.assertEqual(len(unique_indices), C)

    def test_fps_sample_coverage(self):
        """FPS should provide good coverage (points far apart)."""
        B, H, N, d = 1, 1, 100, 64
        C = 10

        # Create points in clusters
        points = torch.randn(B, H, N, d)
        indices = fps_sample(points, C)

        # Gather selected points
        selected = points[0, 0, indices[0, 0]]  # [C, d]

        # Compute pairwise distances
        dists = torch.cdist(selected.unsqueeze(0), selected.unsqueeze(0))[0]

        # Minimum distance should be reasonable (not too small)
        # Set diagonal to large value to ignore self-distances
        dists_no_diag = dists.clone()
        dists_no_diag.fill_diagonal_(1e10)
        min_dist = dists_no_diag.min().item()

        # FPS should select points that are not too close
        self.assertGreater(min_dist, 0.1)

    def test_fps_sample_edge_cases(self):
        """Handle edge cases: C=1, C=N, C>N."""
        B, H, N, d = 2, 4, 100, 64

        points = torch.randn(B, H, N, d)

        # C=1
        indices = fps_sample(points, 1)
        self.assertEqual(indices.shape, (B, H, 1))

        # C=N
        indices = fps_sample(points, N)
        self.assertEqual(indices.shape, (B, H, N))

        # C>N (should return N indices, not C)
        indices = fps_sample(points, N + 10)
        self.assertEqual(indices.shape, (B, H, N))


class TestFPSInitializeCentroids(unittest.TestCase):
    """Tests for fps_initialize_centroids function."""

    def test_fps_init_centroids_shape(self):
        """Centroids must have correct shape."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        K = torch.randn(B, H, N, d)
        centroids = fps_initialize_centroids(K, C)

        self.assertEqual(centroids.shape, (B, H, C, d))

    def test_fps_init_centroids_from_keys(self):
        """Centroids should be actual keys (not interpolated)."""
        B, H, N, d = 1, 1, 100, 64
        C = 32

        K = torch.randn(B, H, N, d)
        centroids = fps_initialize_centroids(K, C)

        # Each centroid should be exactly one of the keys
        # Check by finding the minimum distance from each centroid to all keys
        dists = torch.cdist(centroids, K)  # [B, H, C, N]
        min_dists = dists.min(dim=-1).values  # [B, H, C]

        # All centroids should be exact keys (distance ~0)
        # Use 0.01 threshold to account for floating point precision
        self.assertTrue((min_dists < 0.01).all())


class TestExpertChoiceRouting(unittest.TestCase):
    """Tests for expert_choice_routing function."""

    def test_routing_weights_shape(self):
        """Routing weights must have shape [B, H, N, C]."""
        B, H, N, d = 2, 4, 100, 64
        C = 32
        k = 8

        Q = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)

        routing_weights, expert_mask = expert_choice_routing(Q, centroids, k)

        self.assertEqual(routing_weights.shape, (B, H, N, C))
        self.assertEqual(expert_mask.shape, (B, H, C, k))

    def test_routing_weights_normalized(self):
        """Routing weights should sum to 1 per token."""
        B, H, N, d = 2, 4, 100, 64
        C = 32
        k = 8

        Q = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)

        routing_weights, _ = expert_choice_routing(Q, centroids, k)

        # Sum over experts should be 1
        weight_sums = routing_weights.sum(dim=-1)  # [B, H, N]
        self.assertTrue(
            torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
        )

    def test_expert_mask_valid_indices(self):
        """Expert mask indices should be valid."""
        B, H, N, d = 2, 4, 100, 64
        C = 32
        k = 8

        Q = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)

        _, expert_mask = expert_choice_routing(Q, centroids, k)

        self.assertTrue((expert_mask >= 0).all())
        self.assertTrue((expert_mask < N).all())

    def test_expert_mask_selection_count(self):
        """Each expert should select exactly k tokens."""
        B, H, N, d = 2, 4, 100, 64
        C = 32
        k = 8

        Q = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)

        _, expert_mask = expert_choice_routing(Q, centroids, k)

        self.assertEqual(expert_mask.shape[-1], k)


class TestRoutedAttention(unittest.TestCase):
    """Tests for routed_attention function."""

    def test_routed_attention_shape(self):
        """Output shape must match input."""
        B, H, N, d = 2, 4, 100, 64
        C = 32
        k = 8

        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)

        routing_weights, expert_mask = expert_choice_routing(Q, centroids, k)
        output = routed_attention(Q, K, V, routing_weights, expert_mask, centroids)

        self.assertEqual(output.shape, (B, H, N, d))

    def test_routed_attention_no_nan(self):
        """Output should not contain NaN."""
        B, H, N, d = 2, 4, 100, 64
        C = 32
        k = 8

        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)

        routing_weights, expert_mask = expert_choice_routing(Q, centroids, k)
        output = routed_attention(Q, K, V, routing_weights, expert_mask, centroids)

        self.assertFalse(torch.isnan(output).any())


class TestRoutingMetrics(unittest.TestCase):
    """Tests for routing balance and coverage metrics."""

    def test_routing_balance_shape(self):
        """Balance should have shape [B, H]."""
        B, H, N, C = 2, 4, 100, 32

        routing_weights = torch.softmax(torch.randn(B, H, N, C), dim=-1)
        balance = compute_routing_balance(routing_weights)

        self.assertEqual(balance.shape, (B, H))

    def test_routing_balance_range(self):
        """Balance should be in [0, 1]."""
        B, H, N, C = 2, 4, 100, 32

        routing_weights = torch.softmax(torch.randn(B, H, N, C), dim=-1)
        balance = compute_routing_balance(routing_weights)

        self.assertTrue((balance >= 0).all())
        self.assertTrue((balance <= 1).all())

    def test_coverage_shape(self):
        """Coverage should have shape [B, H]."""
        B, H, C, k = 2, 4, 32, 8
        N = 100

        expert_mask = torch.randint(0, N, (B, H, C, k))
        coverage = compute_coverage(expert_mask, N)

        self.assertEqual(coverage.shape, (B, H))

    def test_coverage_range(self):
        """Coverage should be in [0, 1]."""
        B, H, C, k = 2, 4, 32, 8
        N = 100

        expert_mask = torch.randint(0, N, (B, H, C, k))
        coverage = compute_coverage(expert_mask, N)

        self.assertTrue((coverage >= 0).all())
        self.assertTrue((coverage <= 1).all())


class TestFPSExpertChoiceAttention(unittest.TestCase):
    """Tests for FPSExpertChoiceAttention module."""

    def test_forward_shape(self):
        """Output shape must match input."""
        B, N, d_model = 2, 100, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = FPSExpertChoiceAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=32,
            top_k=8,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_forward_no_nan(self):
        """Output should not contain NaN."""
        B, N, d_model = 2, 100, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = FPSExpertChoiceAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=32,
            top_k=8,
        )

        output, _ = module(x)

        self.assertFalse(torch.isnan(output).any())

    def test_return_routing_info(self):
        """Should return routing info when requested."""
        B, N, d_model = 2, 100, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = FPSExpertChoiceAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=32,
            top_k=8,
        )

        output, routing_info = module(x, return_routing=True)

        self.assertIsNotNone(routing_info)
        self.assertIn("routing_weights", routing_info)
        self.assertIn("expert_mask", routing_info)
        self.assertIn("balance", routing_info)
        self.assertIn("coverage", routing_info)

    def test_fps_vs_learnable_init(self):
        """Both FPS and learnable init should work."""
        B, N, d_model = 2, 100, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)

        # FPS init
        module_fps = FPSExpertChoiceAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=32,
            top_k=8,
            use_fps_init=True,
        )

        # Learnable init
        module_learn = FPSExpertChoiceAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=32,
            top_k=8,
            use_fps_init=False,
        )

        output_fps, _ = module_fps(x)
        output_learn, _ = module_learn(x)

        self.assertEqual(output_fps.shape, (B, N, d_model))
        self.assertEqual(output_learn.shape, (B, N, d_model))


class TestBubbleCentroidsV4(unittest.TestCase):
    """Tests for BubbleCentroidsV4 module."""

    def test_euclidean_centroids_shape(self):
        """Euclidean centroids should have correct shape."""
        B = 2
        num_heads = 8
        num_experts = 32
        head_dim = 64

        centroids_module = BubbleCentroidsV4(
            num_heads=num_heads,
            num_experts=num_experts,
            head_dim=head_dim,
            manifold_type=ManifoldType.EUCLIDEAN,
        )

        centroids = centroids_module(B)

        self.assertEqual(centroids.shape, (B, num_heads, num_experts, head_dim))

    def test_distance_computation_shape(self):
        """Distance computation should return correct shape."""
        B, H, N, d = 2, 8, 100, 64
        C = 32

        centroids_module = BubbleCentroidsV4(
            num_heads=H,
            num_experts=C,
            head_dim=d,
            manifold_type=ManifoldType.EUCLIDEAN,
        )

        Q = torch.randn(B, H, N, d)
        distances = centroids_module.distance_to_queries(Q)

        self.assertEqual(distances.shape, (B, H, N, C))

    def test_poincare_centroids_if_available(self):
        """Poincaré centroids should work if geoopt available."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        B = 2
        num_heads = 8
        num_experts = 32
        head_dim = 64

        centroids_module = BubbleCentroidsV4(
            num_heads=num_heads,
            num_experts=num_experts,
            head_dim=head_dim,
            manifold_type=ManifoldType.POINCARE,
        )

        centroids = centroids_module(B)
        self.assertEqual(centroids.shape, (B, num_heads, num_experts, head_dim))

    def test_sphere_centroids_if_available(self):
        """Sphere centroids should work if geoopt available."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        B = 2
        num_heads = 8
        num_experts = 32
        head_dim = 64

        centroids_module = BubbleCentroidsV4(
            num_heads=num_heads,
            num_experts=num_experts,
            head_dim=head_dim,
            manifold_type=ManifoldType.SPHERE,
        )

        centroids = centroids_module(B)
        self.assertEqual(centroids.shape, (B, num_heads, num_experts, head_dim))


class TestHybridManifoldCentroids(unittest.TestCase):
    """Tests for HybridManifoldCentroids module."""

    def test_hybrid_centroids_shape(self):
        """Hybrid centroids should concatenate correctly."""
        B = 2
        num_heads = 8
        num_experts = 32
        head_dim = 64

        hybrid_module = HybridManifoldCentroids(
            num_heads=num_heads,
            num_experts=num_experts,
            head_dim=head_dim,
            euclidean_heads=4,
            poincare_heads=4,
        )

        centroids = hybrid_module(B)

        self.assertEqual(centroids.shape, (B, num_heads, num_experts, head_dim))

    def test_hybrid_distance_computation(self):
        """Hybrid distance computation should work."""
        B, H, N, d = 2, 8, 100, 64
        C = 32

        hybrid_module = HybridManifoldCentroids(
            num_heads=H,
            num_experts=C,
            head_dim=d,
            euclidean_heads=4,
            poincare_heads=4,
        )

        Q = torch.randn(B, H, N, d)
        distances = hybrid_module.distance_to_queries(Q)

        self.assertEqual(distances.shape, (B, H, N, C))


class TestManifoldFunctions(unittest.TestCase):
    """Tests for manifold utility functions."""

    def test_get_manifold_euclidean(self):
        """Should return Euclidean manifold."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        manifold = get_manifold(ManifoldType.EUCLIDEAN, dim=64)
        self.assertIsNotNone(manifold)

    def test_get_manifold_poincare(self):
        """Should return Poincaré ball manifold."""
        if not GEOOPT_AVAILABLE:
            self.skipTest("geoopt not available")

        manifold = get_manifold(ManifoldType.POINCARE, dim=64, curvature=1.0)
        self.assertIsNotNone(manifold)

    def test_manifold_distance_euclidean(self):
        """Euclidean distance should match torch.norm."""
        x = torch.randn(2, 4, 64)
        y = torch.randn(2, 4, 64)

        dist = manifold_distance(x, y, manifold=None)
        expected = torch.norm(x - y, dim=-1)

        self.assertTrue(torch.allclose(dist, expected, atol=1e-5))

    def test_project_to_manifold_euclidean(self):
        """Euclidean projection should return input unchanged."""
        x = torch.randn(2, 4, 64)
        projected = project_to_manifold(x, manifold=None)

        self.assertTrue(torch.equal(x, projected))


class TestIntegration(unittest.TestCase):
    """Integration tests for V4 components."""

    def test_full_v4_pipeline(self):
        """Test full V4 pipeline: FPS + routing + attention."""
        B, N, d_model = 2, 100, 512
        num_heads = 8
        num_experts = 32
        top_k = 8

        x = torch.randn(B, N, d_model)

        # Create module
        module = FPSExpertChoiceAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
            use_fps_init=True,
        )

        # Forward pass
        output, routing_info = module(x, return_routing=True)

        # Verify output
        self.assertEqual(output.shape, (B, N, d_model))
        self.assertFalse(torch.isnan(output).any())

        # Verify routing info
        self.assertIsNotNone(routing_info)
        self.assertIn("balance", routing_info)
        self.assertIn("coverage", routing_info)

        # Balance and coverage should be reasonable
        balance = routing_info["balance"]
        coverage = routing_info["coverage"]

        self.assertTrue((balance >= 0).all())
        self.assertTrue((balance <= 1).all())
        self.assertTrue((coverage >= 0).all())
        self.assertTrue((coverage <= 1).all())

    def test_v4_with_manifold_centroids(self):
        """Test V4 with manifold-aware centroids."""
        B, N, d_model = 2, 100, 512
        num_heads = 8
        num_experts = 32
        head_dim = d_model // num_heads

        # Create manifold centroids
        centroids_module = BubbleCentroidsV4(
            num_heads=num_heads,
            num_experts=num_experts,
            head_dim=head_dim,
            manifold_type=ManifoldType.EUCLIDEAN,
        )

        # Get centroids
        centroids = centroids_module(B)

        # Create Q, K, V
        Q = torch.randn(B, num_heads, N, head_dim)
        K = torch.randn(B, num_heads, N, head_dim)
        V = torch.randn(B, num_heads, N, head_dim)

        # Routing
        routing_weights, expert_mask = expert_choice_routing(Q, centroids, top_k=8)

        # Attention
        output = routed_attention(Q, K, V, routing_weights, expert_mask, centroids)

        # Verify
        self.assertEqual(output.shape, (B, num_heads, N, head_dim))
        self.assertFalse(torch.isnan(output).any())


class TestPowerDiagrams(unittest.TestCase):
    """Tests for power_diagram_assign function."""

    def test_power_diagram_assign_shape(self):
        """Power diagram assignment must produce correct shape [B, H, N]."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        tokens = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)
        psi = torch.zeros(B, H, C)

        assignments = power_diagram_assign(tokens, centroids, psi, hard=True)

        self.assertEqual(assignments.shape, (B, H, N))

    def test_psi_affects_assignments(self):
        """Psi bias should change assignments."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        tokens = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)

        # Zero psi (standard Voronoi)
        psi_zero = torch.zeros(B, H, C)
        assignments_zero = power_diagram_assign(tokens, centroids, psi_zero, hard=True)

        # Non-zero psi (should bias toward centroids with higher psi)
        psi_biased = torch.randn(B, H, C)
        assignments_biased = power_diagram_assign(
            tokens, centroids, psi_biased, hard=True
        )

        # Assignments should differ when psi is non-zero
        self.assertFalse(torch.equal(assignments_zero, assignments_biased))

    def test_psi_gradient_with_soft_sort(self):
        """Gradient should flow through psi in soft mode."""
        B, H, N, d = 2, 4, 100, 64
        C = 32

        tokens = torch.randn(B, H, N, d, requires_grad=True)
        centroids = torch.randn(B, H, C, d, requires_grad=True)
        psi = torch.randn(B, H, C, requires_grad=True)

        # Soft assignment mode
        soft_assignments = power_diagram_assign(
            tokens, centroids, psi, hard=False, temperature=1.0
        )

        # Compute a loss and backprop
        loss = soft_assignments.sum()
        loss.backward()

        # Gradients should exist for psi
        self.assertIsNotNone(psi.grad)
        self.assertFalse((psi.grad == 0).all())

    def test_edge_case_n_lt_c(self):
        """Should handle N < C gracefully."""
        B, H, N, d = 2, 4, 10, 64
        C = 32  # More centroids than tokens

        tokens = torch.randn(B, H, N, d)
        centroids = torch.randn(B, H, C, d)
        psi = torch.zeros(B, H, C)

        assignments = power_diagram_assign(tokens, centroids, psi, hard=True)

        self.assertEqual(assignments.shape, (B, H, N))
        # All assignments should be valid indices
        self.assertTrue((assignments >= 0).all())
        self.assertTrue((assignments < C).all())


class TestWarmStart(unittest.TestCase):
    """Tests for warm_start_centroids function."""

    def test_warm_start_blending(self):
        """Warm start should blend 0.7 * current + 0.3 * previous."""
        B, H, C, d = 2, 8, 32, 64

        current = torch.randn(B, H, C, d)
        previous = torch.randn(B, H, C, d)
        alpha = 0.7

        blended = warm_start_centroids(current, previous, alpha=alpha)

        expected = alpha * current + (1 - alpha) * previous
        self.assertTrue(torch.allclose(blended, expected, atol=1e-5))

    def test_warm_start_none_previous(self):
        """Should return current unchanged when previous=None."""
        B, H, C, d = 2, 8, 32, 64

        current = torch.randn(B, H, C, d)
        blended = warm_start_centroids(current, None)

        self.assertTrue(torch.equal(blended, current))

    def test_warm_start_shape_mismatch(self):
        """Should raise ValueError on shape mismatch."""
        B, H, C, d = 2, 8, 32, 64

        current = torch.randn(B, H, C, d)
        previous = torch.randn(B, H, C + 1, d)  # Wrong shape

        with self.assertRaises(ValueError):
            warm_start_centroids(current, previous)

    def test_warm_start_manifold_projection(self):
        """Should work with manifold_type parameter."""
        B, H, C, d = 2, 8, 32, 64

        current = torch.randn(B, H, C, d)
        previous = torch.randn(B, H, C, d)

        # Euclidean manifold (no projection)
        blended = warm_start_centroids(current, previous, manifold_type="euclidean")

        self.assertEqual(blended.shape, (B, H, C, d))


class TestSoftSort(unittest.TestCase):
    """Tests for soft_sort function."""

    def test_soft_sort_permutation(self):
        """Soft sort should produce valid soft permutation."""
        B, H, N, C = 2, 4, 100, 32

        x = torch.randn(B, H, N, C)
        sorted_x, perm = soft_sort(x, temperature=1.0)

        # Check shapes
        self.assertEqual(sorted_x.shape, (B, H, N, C))
        self.assertEqual(perm.shape, (B, H, N, C))

        # Permutation should sum to 1 along last dim (valid probability)
        self.assertTrue(
            torch.allclose(perm.sum(dim=-1), torch.ones(B, H, N), atol=1e-5)
        )

    def test_soft_sort_gradient(self):
        """Soft sort should be differentiable."""
        B, H, N, C = 2, 4, 100, 32

        x = torch.randn(B, H, N, C, requires_grad=True)
        sorted_x, perm = soft_sort(x, temperature=1.0)

        loss = sorted_x.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse((x.grad == 0).all())

    def test_soft_sort_temperature(self):
        """Lower temperature should produce harder sorting."""
        B, H, N, C = 2, 4, 10, 5

        x = torch.randn(B, H, N, C)

        # High temperature (softer)
        _, perm_soft = soft_sort(x, temperature=10.0)

        # Low temperature (harder)
        _, perm_hard = soft_sort(x, temperature=0.1)

        # Lower temperature should have lower entropy (more peaked)
        entropy_soft = -(perm_soft * torch.log(perm_soft + 1e-10)).sum(dim=-1).mean()
        entropy_hard = -(perm_hard * torch.log(perm_hard + 1e-10)).sum(dim=-1).mean()

        self.assertLess(entropy_hard, entropy_soft)


if __name__ == "__main__":
    unittest.main(verbosity=2)
