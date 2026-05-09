"""Tests for V4PlateauAdapter — NumPy adapter for Bubble Transformer V4.

Tests cover:
- V4PlateauAdapter initialization (FPS + Expert-Choice)
- Forward pass output shapes and numerical stability
- Routing logic with different top_k values
- Edge cases (fewer experts than tokens, single token, C >= N)
- Standalone helper functions (FPS, Expert-Choice routing, Sinkhorn)
- V4Config
"""

import sys
import os
import unittest

# Add experiments/ and tests/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from v4_adapter import (
    V4PlateauAdapter,
    V4Config,
    fps_sample_numpy,
    fps_initialize_centroids_numpy,
    expert_choice_routing_numpy,
    compute_sinkhorn_attention,
)
from test_helpers import create_mock_embeddings


class TestV4PlateauAdapterInit(unittest.TestCase):
    """Tests for V4PlateauAdapter constructor."""

    def test_init_defaults(self):
        """Default parameters should initialize correctly."""
        adapter = V4PlateauAdapter(d_model=512, num_heads=8)

        self.assertEqual(adapter.d_model, 512)
        self.assertEqual(adapter.num_heads, 8)
        self.assertEqual(adapter.head_dim, 64)
        self.assertEqual(adapter.num_experts, 32)
        self.assertEqual(adapter.top_k, 8)
        self.assertEqual(adapter.epsilon, 0.1)
        self.assertEqual(adapter.tau_iters, 5)
        self.assertTrue(adapter.use_fps_init)

    def test_init_custom_params(self):
        """Custom parameters should be stored correctly."""
        adapter = V4PlateauAdapter(
            d_model=1024,
            num_heads=16,
            num_experts=64,
            top_k=16,
            epsilon=0.001,
            tau_iters=10,
            use_fps_init=False,
            seed=42,
        )

        self.assertEqual(adapter.d_model, 1024)
        self.assertEqual(adapter.num_heads, 16)
        self.assertEqual(adapter.head_dim, 64)
        self.assertEqual(adapter.num_experts, 64)
        self.assertEqual(adapter.top_k, 16)
        self.assertEqual(adapter.epsilon, 0.001)
        self.assertEqual(adapter.tau_iters, 10)
        self.assertFalse(adapter.use_fps_init)

    def test_init_invalid_d_model(self):
        """Should raise AssertionError if d_model not divisible by num_heads."""
        with self.assertRaises(AssertionError):
            V4PlateauAdapter(d_model=512, num_heads=7)

    def test_init_creates_projections(self):
        """Should create W_q, W_k, W_v, W_o weight matrices."""
        adapter = V4PlateauAdapter(d_model=512, num_heads=8)

        self.assertEqual(adapter.W_q.shape, (512, 512))
        self.assertEqual(adapter.W_k.shape, (512, 512))
        self.assertEqual(adapter.W_v.shape, (512, 512))
        self.assertEqual(adapter.W_o.shape, (512, 512))

    def test_init_learnable_centroids(self):
        """Should create learnable centroids buffer."""
        adapter = V4PlateauAdapter(d_model=512, num_heads=8, num_experts=32)

        self.assertEqual(adapter.learnable_centroids.shape, (1, 8, 32, 64))


class TestV4PlateauAdapterForward(unittest.TestCase):
    """Tests for V4PlateauAdapter forward pass."""

    def setUp(self):
        """Set standard dimensions."""
        self.B = 2
        self.N = 32
        self.d_model = 512
        self.num_heads = 8

    def test_forward_output_shape(self):
        """Output shape must be [B, N, d_model]."""
        x_torch = create_mock_embeddings(self.B, self.N, self.d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_experts=16,
            use_fps_init=True,
        )
        output = adapter.forward(x)

        self.assertEqual(output.shape, (self.B, self.N, self.d_model))

    def test_forward_return_attention_shape(self):
        """Attention matrix shape must be [B, H, N, N]."""
        x_torch = create_mock_embeddings(self.B, self.N, self.d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_experts=16,
        )
        output, A = adapter.forward(x, return_attention=True)

        self.assertEqual(output.shape, (self.B, self.N, self.d_model))
        self.assertEqual(A.shape, (self.B, self.num_heads, self.N, self.N))

    def test_forward_no_nan_or_inf(self):
        """Output should not contain NaN or Inf."""
        x_torch = create_mock_embeddings(self.B, self.N, self.d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_experts=16,
        )
        output = adapter.forward(x)

        self.assertFalse(np.isnan(output).any())
        self.assertFalse(np.isinf(output).any())

    def test_forward_without_fps(self):
        """Forward should work with learnable centroids (no FPS)."""
        x_torch = create_mock_embeddings(self.B, self.N, self.d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_experts=16,
            use_fps_init=False,
        )
        output = adapter.forward(x)

        self.assertEqual(output.shape, (self.B, self.N, self.d_model))
        self.assertFalse(np.isnan(output).any())

    def test_forward_attention_is_doubly_stochastic(self):
        """Returned attention should approximate doubly-stochastic property."""
        x_torch = create_mock_embeddings(self.B, self.N, self.d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_experts=16,
        )
        output, A = adapter.forward(x, return_attention=True)

        row_sums = A.sum(axis=-1)
        col_sums = A.sum(axis=-2)

        # Sinkhorn with tau_iters=5 should be close to doubly-stochastic
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-2))
        self.assertTrue(np.allclose(col_sums, 1.0, atol=1e-2))


class TestV4PlateauAdapterRouting(unittest.TestCase):
    """Tests for Expert-Choice routing logic."""

    def setUp(self):
        """Set standard dimensions."""
        self.B = 2
        self.N = 32
        self.d_model = 512
        self.num_heads = 8
        self.num_experts = 16

    def test_routing_weights_shape(self):
        """Routing weights should have shape [B, H, N, C]."""
        x_torch = create_mock_embeddings(self.B, self.N, self.d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
            top_k=8,
        )
        # Access routing internals by running forward (routing is computed inside)
        Q = x @ adapter.W_q
        Q = Q.reshape(self.B, self.N, self.num_heads, adapter.head_dim).transpose(
            0, 2, 1, 3
        )
        K = x @ adapter.W_k
        K = K.reshape(self.B, self.N, self.num_heads, adapter.head_dim).transpose(
            0, 2, 1, 3
        )
        centroids = fps_initialize_centroids_numpy(K, self.num_experts)
        routing_weights, expert_mask = expert_choice_routing_numpy(
            Q, centroids, top_k=8
        )

        self.assertEqual(
            routing_weights.shape, (self.B, self.num_heads, self.N, self.num_experts)
        )
        self.assertEqual(
            expert_mask.shape, (self.B, self.num_heads, self.num_experts, 8)
        )

    def test_routing_top_k_variation(self):
        """Different top_k values should produce correct expert_mask shapes."""
        x_torch = create_mock_embeddings(self.B, self.N, self.d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
        )
        Q = x @ adapter.W_q
        Q = Q.reshape(self.B, self.N, self.num_heads, adapter.head_dim).transpose(
            0, 2, 1, 3
        )
        K = x @ adapter.W_k
        K = K.reshape(self.B, self.N, self.num_heads, adapter.head_dim).transpose(
            0, 2, 1, 3
        )
        centroids = fps_initialize_centroids_numpy(K, self.num_experts)

        for top_k in [1, 4, 8, 16]:
            with self.subTest(top_k=top_k):
                routing_weights, expert_mask = expert_choice_routing_numpy(
                    Q, centroids, top_k=top_k
                )
                expected_k = min(top_k, self.N)
                self.assertEqual(
                    expert_mask.shape,
                    (self.B, self.num_heads, self.num_experts, expected_k),
                )

    def test_routing_weights_are_valid_probability_distribution(self):
        """Routing weights should sum to 1 across experts per token."""
        x_torch = create_mock_embeddings(self.B, self.N, self.d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
        )
        Q = x @ adapter.W_q
        Q = Q.reshape(self.B, self.N, self.num_heads, adapter.head_dim).transpose(
            0, 2, 1, 3
        )
        K = x @ adapter.W_k
        K = K.reshape(self.B, self.N, self.num_heads, adapter.head_dim).transpose(
            0, 2, 1, 3
        )
        centroids = fps_initialize_centroids_numpy(K, self.num_experts)
        routing_weights, _ = expert_choice_routing_numpy(Q, centroids, top_k=8)

        sums = routing_weights.sum(axis=-1)
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-5))
        self.assertTrue(np.all(routing_weights >= 0.0))

    def test_routing_duplicate_assignments(self):
        """Expert mask may contain duplicates when top_k > N or few unique tokens."""
        x_torch = create_mock_embeddings(self.B, self.N, self.d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_experts=self.num_experts,
        )
        Q = x @ adapter.W_q
        Q = Q.reshape(self.B, self.N, self.num_heads, adapter.head_dim).transpose(
            0, 2, 1, 3
        )
        K = x @ adapter.W_k
        K = K.reshape(self.B, self.N, self.num_heads, adapter.head_dim).transpose(
            0, 2, 1, 3
        )
        centroids = fps_initialize_centroids_numpy(K, self.num_experts)
        _, expert_mask = expert_choice_routing_numpy(Q, centroids, top_k=8)

        # expert_mask contains indices; verify they are in valid range
        self.assertTrue(np.all(expert_mask >= 0))
        self.assertTrue(np.all(expert_mask < self.N))


class TestV4PlateauAdapterEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def test_fewer_experts_than_tokens(self):
        """Should handle fewer experts than tokens gracefully."""
        B, N, d_model = 2, 64, 512
        num_heads = 8
        num_experts = 4  # Much fewer than N

        x_torch = create_mock_embeddings(B, N, d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=8,
        )
        output = adapter.forward(x)

        self.assertEqual(output.shape, (B, N, d_model))
        self.assertFalse(np.isnan(output).any())
        self.assertFalse(np.isinf(output).any())

    def test_more_centroids_than_points_fps(self):
        """FPS should handle C == N by returning all indices."""
        B, H, N, d = 2, 4, 8, 64
        points = np.random.randn(B, H, N, d).astype(np.float32)
        C = 8  # C == N

        indices = fps_sample_numpy(points, C)
        self.assertEqual(indices.shape, (B, H, N))

        centroids = fps_initialize_centroids_numpy(points, C)
        self.assertEqual(centroids.shape, (B, H, N, d))

    def test_single_token(self):
        """Should handle N=1."""
        B, N, d_model = 2, 1, 512
        num_heads = 8

        x_torch = create_mock_embeddings(B, N, d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=16,
            top_k=1,
        )
        output = adapter.forward(x)

        self.assertEqual(output.shape, (B, N, d_model))
        self.assertFalse(np.isnan(output).any())

    def test_top_k_larger_than_n(self):
        """top_k > N should be clamped to N."""
        B, N, d_model = 2, 4, 512
        num_heads = 8
        num_experts = 2  # Must be <= N to avoid FPS edge case

        x_torch = create_mock_embeddings(B, N, d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=16,  # Larger than N=4
        )
        output = adapter.forward(x)

        self.assertEqual(output.shape, (B, N, d_model))
        self.assertFalse(np.isnan(output).any())

    def test_small_dimensions(self):
        """Should handle small d_model and few heads."""
        B, N, d_model = 1, 8, 64
        num_heads = 4

        x_torch = create_mock_embeddings(B, N, d_model)
        x = x_torch.numpy().astype(np.float32)

        adapter = V4PlateauAdapter(
            d_model=d_model,
            num_heads=num_heads,
            num_experts=4,
            top_k=2,
        )
        output = adapter.forward(x)

        self.assertEqual(output.shape, (B, N, d_model))


class TestV4AdapterFunctions(unittest.TestCase):
    """Tests for standalone helper functions."""

    def test_fps_sample_numpy_coverage(self):
        """FPS should select C distinct points when C <= N."""
        B, H, N, d = 2, 4, 32, 64
        points = np.random.randn(B, H, N, d).astype(np.float32)
        C = 16

        indices = fps_sample_numpy(points, C)
        self.assertEqual(indices.shape, (B, H, C))

        # All indices should be in valid range
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < N))

    def test_fps_sample_numpy_distinct(self):
        """FPS should select distinct points per batch/head."""
        B, H, N, d = 1, 1, 16, 32
        points = np.random.randn(B, H, N, d).astype(np.float32)
        C = 8

        indices = fps_sample_numpy(points, C)
        # For a single batch/head, all indices should be unique
        unique_indices = np.unique(indices[0, 0])
        self.assertEqual(len(unique_indices), C)

    def test_expert_choice_routing_numpy_output_shapes(self):
        """Expert-Choice routing should return correct shapes."""
        B, H, N, d = 2, 4, 32, 64
        C = 16
        top_k = 8

        Q = np.random.randn(B, H, N, d).astype(np.float32)
        centroids = np.random.randn(B, H, C, d).astype(np.float32)

        routing_weights, expert_mask = expert_choice_routing_numpy(
            Q, centroids, top_k=top_k
        )

        self.assertEqual(routing_weights.shape, (B, H, N, C))
        self.assertEqual(expert_mask.shape, (B, H, C, top_k))

    def test_compute_sinkhorn_attention_shape(self):
        """Sinkhorn attention should return [B, H, N, N]."""
        B, H, N, d = 2, 4, 16, 64

        Q = np.random.randn(B, H, N, d).astype(np.float32)
        K = np.random.randn(B, H, N, d).astype(np.float32)

        A = compute_sinkhorn_attention(Q, K, epsilon=0.1, tau_iters=5)

        self.assertEqual(A.shape, (B, H, N, N))

    def test_compute_sinkhorn_attention_no_nan(self):
        """Sinkhorn attention should not produce NaN or Inf."""
        B, H, N, d = 2, 4, 16, 64

        Q = np.random.randn(B, H, N, d).astype(np.float32)
        K = np.random.randn(B, H, N, d).astype(np.float32)

        A = compute_sinkhorn_attention(Q, K, epsilon=0.01, tau_iters=5)

        self.assertFalse(np.isnan(A).any())
        self.assertFalse(np.isinf(A).any())

    def test_compute_sinkhorn_attention_row_sums(self):
        """Sinkhorn attention rows should approximately sum to 1."""
        B, H, N, d = 2, 4, 16, 64

        Q = np.random.randn(B, H, N, d).astype(np.float32)
        K = np.random.randn(B, H, N, d).astype(np.float32)

        A = compute_sinkhorn_attention(Q, K, epsilon=0.1, tau_iters=5)
        row_sums = A.sum(axis=-1)

        # 5 Sinkhorn iterations give approximate doubly-stochastic property
        self.assertTrue(np.allclose(row_sums, 1.0, atol=0.1))


class TestV4Config(unittest.TestCase):
    """Tests for V4Config dataclass-like object."""

    def test_v4_config_defaults(self):
        """Default config values should match expected defaults."""
        config = V4Config()

        self.assertEqual(config.num_experts, 32)
        self.assertEqual(config.top_k, 8)
        self.assertTrue(config.use_fps_init)
        self.assertEqual(config.epsilon, 0.001)

    def test_v4_config_custom(self):
        """Custom config values should be stored."""
        config = V4Config(
            num_experts=64,
            top_k=16,
            use_fps_init=False,
            epsilon=0.05,
        )

        self.assertEqual(config.num_experts, 64)
        self.assertEqual(config.top_k, 16)
        self.assertFalse(config.use_fps_init)
        self.assertEqual(config.epsilon, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
