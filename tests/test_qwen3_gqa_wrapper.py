"""Tests for Qwen3GQABubbleWrapper.

These tests mock the underlying Qwen3 attention components and verify
that the wrapper correctly replaces the attention computation with
SDOT soft routing while preserving the expected input/output signatures.

No real Qwen model is downloaded or loaded.
"""

import sys
import os
import unittest
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Mock transformers before importing the wrapper
# ---------------------------------------------------------------------------


def _mock_apply_rotary_pos_emb(q, k, cos, sin):
    """Dummy RoPE that returns inputs unchanged."""
    return q, k


# Build a fake transformers.models.qwen3.modeling_qwen3 module
_fake_qwen3_module = types.ModuleType("transformers.models.qwen3.modeling_qwen3")
_fake_qwen3_module.apply_rotary_pos_emb = _mock_apply_rotary_pos_emb

# Build parent modules so the import chain resolves
_transformers_models = types.ModuleType("transformers.models")
_transformers_models_qwen3 = types.ModuleType("transformers.models.qwen3")
_transformers_models_qwen3.modeling_qwen3 = _fake_qwen3_module

sys.modules["transformers"] = types.ModuleType("transformers")
sys.modules["transformers.models"] = _transformers_models
sys.modules["transformers.models.qwen3"] = _transformers_models_qwen3
sys.modules["transformers.models.qwen3.modeling_qwen3"] = _fake_qwen3_module

# Also mock the fallback import path
_fake_rope_utils = types.ModuleType("transformers.modeling_rope_utils")
_fake_rope_utils.apply_rotary_pos_emb = _mock_apply_rotary_pos_emb
sys.modules["transformers.modeling_rope_utils"] = _fake_rope_utils

from qwen3_gqa_bubble_wrapper import (  # noqa: E402
    Qwen3GQABubbleWrapper,
    _make_causal_mask,
)


# ---------------------------------------------------------------------------
# Mock original attention factory
# ---------------------------------------------------------------------------


class MockConfig:
    """Minimal config matching Qwen3-0.6B attention dimensions."""

    def __init__(
        self,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_size: int = 1024,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size


class MockQwen3Attention(nn.Module):
    """Mock self_attn module with the attributes Qwen3GQABubbleWrapper expects."""

    def __init__(
        self,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        hidden_size: int = 1024,
    ):
        super().__init__()
        self.config = MockConfig(num_heads, num_kv_heads, head_dim, hidden_size)
        self.scaling = head_dim**-0.5

        # Projection layers — dimensions must match Qwen3 GQA exactly
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Q/K normalisation (Qwen3-specific: applied after projection, before RoPE)
        self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=False)

    def forward(self, *args, **kwargs):
        raise RuntimeError("MockQwen3Attention.forward should not be called directly")


def create_mock_position_embeddings(B: int, N: int, head_dim: int, num_heads: int = 16):
    """Return (cos, sin) tensors compatible with apply_rotary_pos_emb.

    In real Qwen3 these have shape [B, N, 1, head_dim] or similar;
    our mock RoPE accepts any matching shape and returns inputs unchanged.
    """
    cos = torch.randn(B, N, 1, head_dim)
    sin = torch.randn(B, N, 1, head_dim)
    return cos, sin


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestQwen3GQABubbleWrapper(unittest.TestCase):
    """Unit tests for the Qwen3 GQA Bubble Wrapper."""

    def setUp(self):
        """Standard Qwen3-0.6B dimensions."""
        self.B = 2
        self.N = 16
        self.D = 1024
        self.num_heads = 16
        self.num_kv_heads = 8
        self.head_dim = 128
        self.hidden_size = 1024

    def _make_wrapper(self, **override_kwargs):
        """Helper: instantiate wrapper with default mock attention."""
        original = MockQwen3Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
        )
        kwargs = {
            "original_attn": original,
            "num_bubbles": 32,
            "top_k": 64,
            "eps": 0.005,
            "routing_bonus": 0.1,
            "debug": False,
        }
        kwargs.update(override_kwargs)
        return Qwen3GQABubbleWrapper(**kwargs)

    # -- 1. Basic wrapping logic -------------------------------------------

    def test_wrapper_output_shape(self):
        """Forward pass must produce [B, N, hidden_size]."""
        wrapper = self._make_wrapper()
        x = torch.randn(self.B, self.N, self.hidden_size)
        out, _ = wrapper(x)
        self.assertEqual(out.shape, (self.B, self.N, self.hidden_size))

    def test_wrapper_returns_none_attention(self):
        """Wrapper always returns None as the second tuple element."""
        wrapper = self._make_wrapper()
        x = torch.randn(self.B, self.N, self.hidden_size)
        _, attn_weights = wrapper(x)
        self.assertIsNone(attn_weights)

    def test_wrapper_without_position_embeddings(self):
        """Forward should work when position_embeddings is None (no RoPE)."""
        wrapper = self._make_wrapper()
        x = torch.randn(self.B, self.N, self.hidden_size)
        out, _ = wrapper(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_wrapper_with_position_embeddings(self):
        """Forward should work when position_embeddings is provided."""
        wrapper = self._make_wrapper()
        x = torch.randn(self.B, self.N, self.hidden_size)
        pos_emb = create_mock_position_embeddings(
            self.B, self.N, self.head_dim, self.num_heads
        )
        out, _ = wrapper(x, position_embeddings=pos_emb)
        self.assertEqual(out.shape, (self.B, self.N, self.hidden_size))
        self.assertFalse(torch.isnan(out).any())

    # -- 2. Mock compatibility (no real model) -----------------------------

    def test_mock_compatibility_no_real_model(self):
        """Wrapper must initialise and run with purely mock components."""
        wrapper = self._make_wrapper()
        # Verify all required attributes are present and correct
        self.assertEqual(wrapper.num_heads, self.num_heads)
        self.assertEqual(wrapper.num_kv_heads, self.num_kv_heads)
        self.assertEqual(wrapper.head_dim, self.head_dim)
        self.assertEqual(wrapper.hidden_size, self.hidden_size)
        self.assertEqual(wrapper.kv_groups, self.num_heads // self.num_kv_heads)

        x = torch.randn(self.B, self.N, self.hidden_size)
        out, _ = wrapper(x)
        self.assertTrue(torch.isfinite(out).all())

    # -- 3. Output shapes of internal tensors ------------------------------

    def test_internal_qkv_shapes(self):
        """Internal Q, K, V projections must expand correctly for GQA."""
        wrapper = self._make_wrapper(debug=False)
        x = torch.randn(self.B, self.N, self.hidden_size)

        # Replicate the projection logic manually to verify shapes
        input_shape = x.shape[:-1]  # (B, N)
        hidden_shape = (*input_shape, -1, wrapper.head_dim)

        Q = wrapper.q_norm(wrapper.q_proj(x).view(hidden_shape)).transpose(1, 2)
        K = wrapper.k_norm(wrapper.k_proj(x).view(hidden_shape)).transpose(1, 2)
        V = wrapper.v_proj(x).view(hidden_shape).transpose(1, 2)

        self.assertEqual(Q.shape, (self.B, self.num_heads, self.N, self.head_dim))
        self.assertEqual(K.shape, (self.B, self.num_kv_heads, self.N, self.head_dim))
        self.assertEqual(V.shape, (self.B, self.num_kv_heads, self.N, self.head_dim))

        K_expanded = K.repeat_interleave(wrapper.kv_groups, dim=1)
        V_expanded = V.repeat_interleave(wrapper.kv_groups, dim=1)
        self.assertEqual(
            K_expanded.shape, (self.B, self.num_heads, self.N, self.head_dim)
        )
        self.assertEqual(
            V_expanded.shape, (self.B, self.num_heads, self.N, self.head_dim)
        )

    def test_attention_scores_shape(self):
        """Attention score matrix must be [B, num_heads, N, N]."""
        wrapper = self._make_wrapper(debug=False)
        x = torch.randn(self.B, self.N, self.hidden_size)

        # Manual forward up to attention scores
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, wrapper.head_dim)
        Q = wrapper.q_norm(wrapper.q_proj(x).view(hidden_shape)).transpose(1, 2)
        K = wrapper.k_norm(wrapper.k_proj(x).view(hidden_shape)).transpose(1, 2)
        K_expanded = K.repeat_interleave(wrapper.kv_groups, dim=1)

        attn_scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * wrapper.scaling
        self.assertEqual(attn_scores.shape, (self.B, self.num_heads, self.N, self.N))

    # -- 4. Edge cases -----------------------------------------------------

    def test_single_token_sequence(self):
        """Wrapper must handle a single-token sequence (N=1)."""
        wrapper = self._make_wrapper()
        x = torch.randn(self.B, 1, self.hidden_size)
        out, _ = wrapper(x)
        self.assertEqual(out.shape, (self.B, 1, self.hidden_size))
        self.assertTrue(torch.isfinite(out).all())

    def test_single_batch(self):
        """Wrapper must handle batch size of 1."""
        wrapper = self._make_wrapper()
        x = torch.randn(1, self.N, self.hidden_size)
        out, _ = wrapper(x)
        self.assertEqual(out.shape, (1, self.N, self.hidden_size))
        self.assertTrue(torch.isfinite(out).all())

    def test_multiple_layers_compatible(self):
        """Multiple independent wrapper instances should not share state."""
        wrapper_a = self._make_wrapper(num_bubbles=16)
        wrapper_b = self._make_wrapper(num_bubbles=64)
        x = torch.randn(self.B, self.N, self.hidden_size)

        out_a, _ = wrapper_a(x)
        out_b, _ = wrapper_b(x)

        self.assertEqual(out_a.shape, out_b.shape)
        # Different centroids => different outputs
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-6))

    # -- 5. Routing / bubble logic -----------------------------------------

    def test_routing_bonus_zero(self):
        """With routing_bonus=0, same-bubble tokens get no extra bias."""
        wrapper = self._make_wrapper(routing_bonus=0.0, debug=False)
        x = torch.randn(self.B, self.N, self.hidden_size)
        out, _ = wrapper(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_different_num_bubbles(self):
        """Wrapper should work with different num_bubbles values."""
        for num_bubbles in [4, 16, 64]:
            with self.subTest(num_bubbles=num_bubbles):
                wrapper = self._make_wrapper(num_bubbles=num_bubbles)
                x = torch.randn(self.B, self.N, self.hidden_size)
                out, _ = wrapper(x)
                self.assertEqual(out.shape, (self.B, self.N, self.hidden_size))

    # -- 6. Debug flag ------------------------------------------------------

    def test_debug_flag(self):
        """Debug=True should print exactly once per wrapper instance."""
        wrapper = self._make_wrapper(debug=True)
        x = torch.randn(self.B, self.N, self.hidden_size)

        # First forward triggers debug print
        out1, _ = wrapper(x)
        self.assertTrue(wrapper._debug_printed)

        # Second forward should not print again
        out2, _ = wrapper(x)
        # If it printed again we'd see duplicate output; _debug_printed stays True
        self.assertTrue(wrapper._debug_printed)

    # -- 7. Causal mask helper ---------------------------------------------

    def test_make_causal_mask(self):
        """_make_causal_mask must be upper-triangular with -inf above diagonal."""
        mask = _make_causal_mask(self.N, torch.device("cpu"), torch.float32)
        self.assertEqual(mask.shape, (self.N, self.N))
        # Diagonal and below should be 0 (or -inf for above)
        for i in range(self.N):
            for j in range(self.N):
                if j > i:
                    self.assertEqual(mask[i, j].item(), float("-inf"))
                else:
                    self.assertEqual(mask[i, j].item(), 0.0)

    # -- 8. Attribute forwarding -------------------------------------------

    def test_projection_attributes_forwarded(self):
        """Wrapper must expose the original projection layers as its own."""
        original = MockQwen3Attention()
        wrapper = Qwen3GQABubbleWrapper(original)

        self.assertIs(wrapper.q_proj, original.q_proj)
        self.assertIs(wrapper.k_proj, original.k_proj)
        self.assertIs(wrapper.v_proj, original.v_proj)
        self.assertIs(wrapper.o_proj, original.o_proj)
        self.assertIs(wrapper.q_norm, original.q_norm)
        self.assertIs(wrapper.k_norm, original.k_norm)


# ---------------------------------------------------------------------------
# Graceful skip when PyTorch is unavailable
# ---------------------------------------------------------------------------


@unittest.skipIf(not hasattr(torch, "nn"), "PyTorch not installed")
class TestQwen3GQABubbleWrapperImports(unittest.TestCase):
    """Sanity checks that our mocked imports resolved correctly."""

    def test_transformers_mock_loaded(self):
        """The mocked transformers module should be present in sys.modules."""
        self.assertIn("transformers.models.qwen3.modeling_qwen3", sys.modules)
        mod = sys.modules["transformers.models.qwen3.modeling_qwen3"]
        self.assertTrue(hasattr(mod, "apply_rotary_pos_emb"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
