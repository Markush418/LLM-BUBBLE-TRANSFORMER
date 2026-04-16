"""Tests for SDOTAttentionV4 — Bubble Transformer V4
=================================================

Comprehensive unit tests for SDOTAttentionV4 class, including:
- Constructor and initialization
- Forward pass with various configurations
- Expert-Choice routing
- Power Diagrams integration
- API compatibility with V3
"""

import sys
import os
import unittest
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.sdot_attention_v4 import SDOTAttentionV4
from models.sdot_attention import SDOTAttention


class TestSDOTAttentionV4Init(unittest.TestCase):
    """Tests for SDOTAttentionV4 constructor."""

    def test_constructor_defaults(self):
        """Default parameters should work."""
        d_model = 512
        num_heads = 8

        module = SDOTAttentionV4(d_model=d_model, num_heads=num_heads)

        self.assertEqual(module.d_model, d_model)
        self.assertEqual(module.num_heads, num_heads)
        self.assertEqual(module.head_dim, d_model // num_heads)
        self.assertEqual(module.num_centroids, 32)
        self.assertTrue(module.use_baroreceptor)
        self.assertTrue(module.use_fps_init)
        self.assertFalse(module.use_power_diagrams)
        self.assertTrue(module.use_expert_routing)
        self.assertEqual(module.manifold_type, "euclidean")

    def test_constructor_invalid_d_model(self):
        """Should raise AssertionError if d_model not divisible by num_heads."""
        with self.assertRaises(AssertionError):
            SDOTAttentionV4(d_model=512, num_heads=7)  # 512 % 7 != 0

    def test_constructor_parameters(self):
        """All parameters should be accepted and stored."""
        d_model = 512
        num_heads = 8
        num_centroids = 64
        min_C = 32
        max_C = 256
        top_k = 16
        temperature = 0.5
        warm_start_alpha = 0.8

        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_baroreceptor=False,
            use_fps_init=False,
            use_power_diagrams=True,
            use_expert_routing=False,
            manifold_type="poincare",
            min_C=min_C,
            max_C=max_C,
            top_k=top_k,
            temperature=temperature,
            warm_start_alpha=warm_start_alpha,
        )

        self.assertEqual(module.num_centroids, num_centroids)
        self.assertFalse(module.use_baroreceptor)
        self.assertFalse(module.use_fps_init)
        self.assertTrue(module.use_power_diagrams)
        self.assertFalse(module.use_expert_routing)
        self.assertEqual(module.manifold_type, "poincare")
        self.assertEqual(module.min_C, min_C)
        self.assertEqual(module.max_C, max_C)
        self.assertEqual(module.top_k, top_k)
        self.assertEqual(module.temperature, temperature)
        self.assertEqual(module.warm_start_alpha, warm_start_alpha)

    def test_constructor_creates_projections(self):
        """Should create W_q, W_k, W_v, W_o projections."""
        module = SDOTAttentionV4(d_model=512, num_heads=8)

        self.assertIsInstance(module.W_q, torch.nn.Linear)
        self.assertIsInstance(module.W_k, torch.nn.Linear)
        self.assertIsInstance(module.W_v, torch.nn.Linear)
        self.assertIsInstance(module.W_o, torch.nn.Linear)

    def test_constructor_baroreceptor(self):
        """Should create baroreceptor when use_baroreceptor=True."""
        module = SDOTAttentionV4(d_model=512, num_heads=8, use_baroreceptor=True)

        self.assertTrue(hasattr(module, "baroreceptor"))

    def test_constructor_no_baroreceptor(self):
        """Should not create baroreceptor when use_baroreceptor=False."""
        module = SDOTAttentionV4(d_model=512, num_heads=8, use_baroreceptor=False)

        self.assertFalse(hasattr(module, "baroreceptor"))

    def test_constructor_power_diagrams_psi(self):
        """Should create psi parameter when use_power_diagrams=True."""
        num_heads = 8
        num_centroids = 32

        module = SDOTAttentionV4(
            d_model=512,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_power_diagrams=True,
        )

        self.assertIsNotNone(module.psi)
        self.assertEqual(module.psi.shape, (1, num_heads, num_centroids))
        # Should be initialized to zeros
        self.assertTrue(torch.allclose(module.psi, torch.zeros_like(module.psi)))

    def test_constructor_no_power_diagrams_psi(self):
        """Should not create psi when use_power_diagrams=False."""
        module = SDOTAttentionV4(d_model=512, num_heads=8, use_power_diagrams=False)

        self.assertIsNone(module.psi)

    def test_constructor_learnable_centroids(self):
        """Should create bubble_centroids when use_fps_init=False."""
        module = SDOTAttentionV4(
            d_model=512,
            num_heads=8,
            num_centroids=32,
            use_fps_init=False,
        )

        self.assertTrue(hasattr(module, "bubble_centroids"))

    def test_constructor_no_learnable_centroids(self):
        """Should not create bubble_centroids when use_fps_init=True."""
        module = SDOTAttentionV4(
            d_model=512,
            num_heads=8,
            num_centroids=32,
            use_fps_init=True,
        )

        self.assertFalse(hasattr(module, "bubble_centroids"))


class TestSDOTAttentionV4Forward(unittest.TestCase):
    """Tests for SDOTAttentionV4 forward pass."""

    def test_forward_shape(self):
        """Output shape must match input [B, N, d_model]."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_forward_no_nan(self):
        """Output should not contain NaN."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, _ = module(x)

        self.assertFalse(torch.isnan(output).any())

    def test_forward_gradient_flow(self):
        """Gradients should flow through the module."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model, requires_grad=True)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, _ = module(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse((x.grad == 0).all())

    def test_forward_with_attention_mask(self):
        """Should handle attention_mask parameter (even if not used)."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        attention_mask = torch.ones(B, N)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, _ = module(x, attention_mask=attention_mask)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_forward_return_assignments(self):
        """Should return assignments when requested."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, assignments = module(x, return_assignments=True)

        self.assertIsNotNone(assignments)
        self.assertIn("centroids", assignments)
        self.assertIn("balance", assignments)
        self.assertIn("coverage", assignments)

    def test_forward_warm_start(self):
        """Warm-start from previous centroids should work."""
        B, N, d_model = 2, 128, 512
        num_heads = 8
        num_centroids = 32
        head_dim = d_model // num_heads

        x = torch.randn(B, N, d_model)
        previous_centroids = torch.randn(B, num_heads, num_centroids, head_dim)

        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_baroreceptor=False,
        )

        output, assignments = module(
            x, return_assignments=True, previous_centroids=previous_centroids
        )

        self.assertEqual(output.shape, (B, N, d_model))
        self.assertIsNotNone(assignments)

    def test_forward_with_baroreceptor(self):
        """Should work with baroreceptor (dynamic C)."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            use_baroreceptor=True,
            min_C=16,
            max_C=128,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_forward_with_fixed_C(self):
        """forward_with_fixed_C should work."""
        B, N, d_model = 2, 128, 512
        num_heads = 8
        C = 64

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            use_baroreceptor=True,
            min_C=16,
            max_C=128,
        )

        output, _ = module.forward_with_fixed_C(x, C=C)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_forward_edge_case_n_lt_c(self):
        """Should handle N < C gracefully."""
        B, N, d_model = 2, 10, 512
        num_heads = 8
        num_centroids = 32  # More centroids than tokens

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_baroreceptor=False,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))
        self.assertFalse(torch.isnan(output).any())


class TestSDOTAttentionV4Routing(unittest.TestCase):
    """Tests for Expert-Choice routing."""

    def test_expert_routing_enabled(self):
        """Expert-Choice routing should work when enabled."""
        B, N, d_model = 2, 128, 512
        num_heads = 8
        num_centroids = 32
        top_k = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_baroreceptor=False,
            use_expert_routing=True,
            top_k=top_k,
        )

        output, assignments = module(x, return_assignments=True)

        self.assertIsNotNone(assignments)
        self.assertIn("routing_weights", assignments)
        self.assertIn("expert_mask", assignments)
        self.assertEqual(
            assignments["routing_weights"].shape, (B, num_heads, N, num_centroids)
        )
        self.assertEqual(
            assignments["expert_mask"].shape, (B, num_heads, num_centroids, top_k)
        )

    def test_expert_routing_disabled(self):
        """Should work without Expert-Choice routing."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            use_expert_routing=False,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))
        self.assertFalse(torch.isnan(output).any())

    def test_routing_balance(self):
        """Routing should be reasonably balanced."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            use_expert_routing=True,
        )

        output, assignments = module(x, return_assignments=True)

        balance = assignments["balance"]
        # Balance should be in [0, 1]
        self.assertTrue((balance >= 0).all())
        self.assertTrue((balance <= 1).all())
        # Balance should be reasonable (not too low)
        self.assertGreater(balance.mean().item(), 0.1)

    def test_routing_coverage(self):
        """Routing should provide good coverage."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            use_expert_routing=True,
        )

        output, assignments = module(x, return_assignments=True)

        coverage = assignments["coverage"]
        # Coverage should be in [0, 1]
        self.assertTrue((coverage >= 0).all())
        self.assertTrue((coverage <= 1).all())


class TestPowerDiagramsIntegration(unittest.TestCase):
    """Tests for Power Diagrams mode."""

    def test_power_diagrams_mode(self):
        """Power Diagrams mode should produce valid output."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            use_power_diagrams=True,
        )

        output, assignments = module(x, return_assignments=True)

        self.assertEqual(output.shape, (B, N, d_model))
        self.assertIn("power_diagram_assignments", assignments)
        self.assertIn("psi", assignments)

    def test_psi_learnable(self):
        """psi should be learnable and have gradient."""
        num_heads = 8
        num_centroids = 32

        module = SDOTAttentionV4(
            d_model=512,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_power_diagrams=True,
        )

        # psi should be a Parameter
        self.assertIsInstance(module.psi, torch.nn.Parameter)

        # psi should have correct shape
        self.assertEqual(module.psi.shape, (1, num_heads, num_centroids))

        # psi should be initialized to zeros
        self.assertTrue(torch.allclose(module.psi, torch.zeros_like(module.psi)))

    def test_psi_affects_output(self):
        """Non-zero psi should affect the output."""
        B, N, d_model = 2, 128, 512
        num_heads = 8
        num_centroids = 32

        x = torch.randn(B, N, d_model)

        # Module with zero psi
        module_zero = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_baroreceptor=False,
            use_power_diagrams=True,
        )

        # Module with non-zero psi
        module_nonzero = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_baroreceptor=False,
            use_power_diagrams=True,
        )
        # Set psi to non-zero values
        with torch.no_grad():
            module_nonzero.psi.fill_(0.5)

        output_zero, _ = module_zero(x)
        output_nonzero, _ = module_nonzero(x)

        # Outputs should differ
        self.assertFalse(torch.allclose(output_zero, output_nonzero, atol=1e-5))

    def test_power_diagrams_vs_expert_choice(self):
        """Both Power Diagrams and Expert-Choice should work together."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            use_power_diagrams=True,
            use_expert_routing=True,
        )

        output, assignments = module(x, return_assignments=True)

        self.assertEqual(output.shape, (B, N, d_model))
        # Should have both power diagram and routing info
        self.assertIn("power_diagram_assignments", assignments)
        self.assertIn("routing_weights", assignments)

    def test_power_diagrams_hard_vs_soft(self):
        """Power Diagrams should work in both hard and soft modes."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)

        # Training mode (soft)
        module_train = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            use_power_diagrams=True,
        )
        module_train.train()

        output_train, _ = module_train(x)

        # Eval mode (hard)
        module_eval = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            use_power_diagrams=True,
        )
        module_eval.eval()

        output_eval, _ = module_eval(x)

        # Both should produce valid outputs
        self.assertEqual(output_train.shape, (B, N, d_model))
        self.assertEqual(output_eval.shape, (B, N, d_model))


class TestAPICompatibility(unittest.TestCase):
    """Tests for API compatibility with V3 SDOTAttention."""

    def test_api_matches_v3(self):
        """API signature should match V3 SDOTAttention."""
        d_model = 512
        num_heads = 8

        # V3
        v3 = SDOTAttention(d_model=d_model, num_heads=num_heads)

        # V4
        v4 = SDOTAttentionV4(d_model=d_model, num_heads=num_heads)

        # Both should have same core attributes
        self.assertEqual(v3.d_model, v4.d_model)
        self.assertEqual(v3.num_heads, v4.num_heads)
        self.assertEqual(v3.head_dim, v4.head_dim)

        # Both should have same projections
        self.assertIsInstance(v4.W_q, torch.nn.Linear)
        self.assertIsInstance(v4.W_k, torch.nn.Linear)
        self.assertIsInstance(v4.W_v, torch.nn.Linear)
        self.assertIsInstance(v4.W_o, torch.nn.Linear)

    def test_drop_in_replacement(self):
        """V4 should be a drop-in replacement for V3."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)

        # V3
        v3 = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        # V4
        v4 = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        # Both should accept same input
        output_v3, _ = v3(x)
        output_v4, _ = v4(x)

        # Both should produce same output shape
        self.assertEqual(output_v3.shape, output_v4.shape)
        self.assertEqual(output_v4.shape, (B, N, d_model))

    def test_forward_with_fixed_c_compatibility(self):
        """forward_with_fixed_C should have same signature as V3."""
        B, N, d_model = 2, 128, 512
        num_heads = 8
        C = 64

        x = torch.randn(B, N, d_model)

        # V3
        v3 = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_baroreceptor=True,
        )

        # V4
        v4 = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            use_baroreceptor=True,
        )

        # Both should accept same call
        output_v3, _ = v3.forward_with_fixed_C(x, C=C)
        output_v4, _ = v4.forward_with_fixed_C(x, C=C)

        # Both should produce same output shape
        self.assertEqual(output_v3.shape, output_v4.shape)

    def test_return_assignments_compatibility(self):
        """return_assignments parameter should work same as V3."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)

        # V4
        v4 = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        # Without assignments
        output_no_assign, none_assign = v4(x, return_assignments=False)
        self.assertIsNone(none_assign)

        # With assignments
        output_assign, assignments = v4(x, return_assignments=True)
        self.assertIsNotNone(assignments)


class TestManifoldTypes(unittest.TestCase):
    """Tests for different manifold types."""

    def test_euclidean_manifold(self):
        """Euclidean manifold should work."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            manifold_type="euclidean",
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_poincare_manifold(self):
        """Poincaré manifold should work."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            manifold_type="poincare",
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_sphere_manifold(self):
        """Sphere manifold should work."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            manifold_type="sphere",
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_stiefel_manifold(self):
        """Stiefel manifold should work."""
        B, N, d_model = 2, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
            manifold_type="stiefel",
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))


class TestGQASupport(unittest.TestCase):
    """Tests for GQA (Grouped Query Attention) support."""

    def test_gqa_constructor(self):
        """Constructor should accept num_kv_heads."""
        model = SDOTAttentionV4(d_model=1024, num_heads=16, num_kv_heads=8)
        self.assertEqual(model.num_heads, 16)
        self.assertEqual(model.num_kv_heads, 8)
        self.assertEqual(model.head_repetition, 2)

    def test_gqa_projection_shapes(self):
        """K, V projections should have correct shapes."""
        model = SDOTAttentionV4(d_model=1024, num_heads=16, num_kv_heads=8)
        # W_q: [num_heads * head_dim, d_model] = [1024, 1024]
        self.assertEqual(model.W_q.weight.shape, (1024, 1024))
        # W_k: [num_kv_heads * head_dim, d_model] = [512, 1024]
        self.assertEqual(model.W_k.weight.shape, (512, 1024))
        # W_v: [num_kv_heads * head_dim, d_model] = [512, 1024]
        self.assertEqual(model.W_v.weight.shape, (512, 1024))
        # W_o: [d_model, num_heads * head_dim] = [1024, 1024]
        self.assertEqual(model.W_o.weight.shape, (1024, 1024))

    def test_gqa_forward_pass(self):
        """Forward pass should work with GQA."""
        model = SDOTAttentionV4(
            d_model=1024, num_heads=16, num_kv_heads=8, use_baroreceptor=False
        )
        x = torch.randn(2, 100, 1024)
        output, _ = model(x)
        self.assertEqual(output.shape, (2, 100, 1024))
        self.assertFalse(torch.isnan(output).any())

    def test_gqa_expansion_correct(self):
        """K, V should be expanded correctly to match Q heads."""
        model = SDOTAttentionV4(
            d_model=512, num_heads=8, num_kv_heads=4, use_baroreceptor=False
        )
        x = torch.randn(1, 10, 512)
        output, _ = model(x)
        # Verify output shape
        self.assertEqual(output.shape, (1, 10, 512))

    def test_gqa_backward_compat(self):
        """num_kv_heads=None should work like before (MHA)."""
        # Old-style construction (no GQA)
        model_old = SDOTAttentionV4(d_model=512, num_heads=8, use_baroreceptor=False)
        # New-style with explicit None
        model_new = SDOTAttentionV4(
            d_model=512, num_heads=8, num_kv_heads=None, use_baroreceptor=False
        )
        x = torch.randn(2, 100, 512)
        output_old, _ = model_old(x)
        output_new, _ = model_new(x)
        # Both should have same shape
        self.assertEqual(output_old.shape, output_new.shape)
        self.assertEqual(output_new.shape, (2, 100, 512))

    def test_gqa_extreme_case(self):
        """Should handle num_kv_heads=1 (maximum sharing, MQA)."""
        model = SDOTAttentionV4(
            d_model=512, num_heads=8, num_kv_heads=1, use_baroreceptor=False
        )
        x = torch.randn(2, 100, 512)
        output, _ = model(x)
        self.assertEqual(output.shape, (2, 100, 512))
        self.assertFalse(torch.isnan(output).any())

    def test_gqa_invalid_config(self):
        """Should raise AssertionError if num_heads not divisible by num_kv_heads."""
        with self.assertRaises(AssertionError):
            SDOTAttentionV4(d_model=512, num_heads=8, num_kv_heads=3)  # 8 % 3 != 0

    def test_gqa_gradient_flow(self):
        """Gradients should flow through GQA expansion."""
        model = SDOTAttentionV4(
            d_model=512, num_heads=8, num_kv_heads=4, use_baroreceptor=False
        )
        x = torch.randn(2, 100, 512, requires_grad=True)
        output, _ = model(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse((x.grad == 0).all())

    def test_gqa_with_expert_routing(self):
        """GQA should work with Expert-Choice routing."""
        model = SDOTAttentionV4(
            d_model=512,
            num_heads=8,
            num_kv_heads=4,
            use_baroreceptor=False,
            use_expert_routing=True,
        )
        x = torch.randn(2, 100, 512)
        output, assignments = model(x, return_assignments=True)
        self.assertEqual(output.shape, (2, 100, 512))
        self.assertIsNotNone(assignments)
        self.assertIn("routing_weights", assignments)

    def test_gqa_with_power_diagrams(self):
        """GQA should work with Power Diagrams."""
        model = SDOTAttentionV4(
            d_model=512,
            num_heads=8,
            num_kv_heads=4,
            use_baroreceptor=False,
            use_power_diagrams=True,
        )
        x = torch.randn(2, 100, 512)
        output, assignments = model(x, return_assignments=True)
        self.assertEqual(output.shape, (2, 100, 512))
        self.assertIn("power_diagram_assignments", assignments)

    def test_gqa_qwen3_config(self):
        """Should match Qwen3-0.6B configuration (16 Q heads, 8 KV heads)."""
        model = SDOTAttentionV4(
            d_model=1024,
            num_heads=16,
            num_kv_heads=8,
            use_baroreceptor=False,
        )
        self.assertEqual(model.num_heads, 16)
        self.assertEqual(model.num_kv_heads, 8)
        self.assertEqual(model.head_dim, 64)
        self.assertEqual(model.head_repetition, 2)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_single_token(self):
        """Should handle N=1."""
        B, N, d_model = 2, 1, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_large_batch(self):
        """Should handle large batch size."""
        B, N, d_model = 32, 128, 512
        num_heads = 8

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_small_d_model(self):
        """Should handle small d_model."""
        B, N, d_model = 2, 128, 64
        num_heads = 4

        x = torch.randn(B, N, d_model)
        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=16,
            use_baroreceptor=False,
        )

        output, _ = module(x)

        self.assertEqual(output.shape, (B, N, d_model))

    def test_previous_centroids_shape_mismatch(self):
        """Should raise ValueError on previous_centroids with different C."""
        B, N, d_model = 2, 128, 512
        num_heads = 8
        head_dim = d_model // num_heads

        x = torch.randn(B, N, d_model)
        # Previous centroids with different C
        previous_centroids = torch.randn(B, num_heads, 16, head_dim)

        module = SDOTAttentionV4(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        # Should raise ValueError (warm_start validates shape)
        with self.assertRaises(ValueError):
            module(x, previous_centroids=previous_centroids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
