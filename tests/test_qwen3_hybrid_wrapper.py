"""Tests for Qwen3HybridGQABubbleWrapper (real Qwen3-0.6B model only)."""

import sys
import os
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import torch

warnings.filterwarnings("ignore")

try:
    from transformers import AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@unittest.skipUnless(
    HAS_TRANSFORMERS
    and torch.cuda.is_available()
    and os.environ.get("RUN_QWEN3_TESTS") == "1",
    "Requires transformers + CUDA + RUN_QWEN3_TESTS=1 (opt-in: downloads Qwen3-0.6B)",
)
class TestQwen3RealModel(unittest.TestCase):
    """Tests with the real Qwen3-0.6B model."""

    @classmethod
    def setUpClass(cls):
        cls.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base",
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        cls.model.eval()

    def test_wrapper_loads_with_real_model(self):
        """Wrapper should load without errors on real Qwen3."""
        from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

        original_attn = self.model.model.layers[0].self_attn
        wrapper = Qwen3HybridGQABubbleWrapper(
            original_attn=original_attn,
            epsilon=0.01,
            lam=0.5,
        ).cuda()

        B, N = 1, 32
        hidden = torch.randn(
            B, N, self.model.config.hidden_size, dtype=torch.float16, device="cuda"
        )
        pos = torch.arange(N, device="cuda").unsqueeze(0)

        with torch.no_grad():
            cos, sin = self.model.model.rotary_emb(hidden, pos)
            out, _ = wrapper(hidden, position_embeddings=(cos, sin))

        self.assertEqual(out.shape, (B, N, self.model.config.hidden_size))
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.isfinite(out).all())

    def test_wrapper_lambda_extremes_real(self):
        """lambda=0 (pure SIRI) and lambda=1 (pure DeltaNet) should differ."""
        from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

        B, N = 1, 16
        hidden = torch.randn(
            B, N, self.model.config.hidden_size, dtype=torch.float16, device="cuda"
        )
        pos = torch.arange(N, device="cuda").unsqueeze(0)

        original_attn = self.model.model.layers[0].self_attn

        with torch.no_grad():
            cos, sin = self.model.model.rotary_emb(hidden, pos)

            out_siri, _ = Qwen3HybridGQABubbleWrapper(
                original_attn=original_attn, epsilon=0.5, lam=0.0,
            ).cuda()(hidden, position_embeddings=(cos, sin))

            out_dn, _ = Qwen3HybridGQABubbleWrapper(
                original_attn=original_attn, epsilon=0.5, lam=1.0,
            ).cuda()(hidden, position_embeddings=(cos, sin))

        self.assertFalse(torch.allclose(out_siri, out_dn, atol=1e-2))

    def test_wrapper_no_nan_real(self):
        """Various epsilon values should not produce NaN on real model."""
        from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

        original_attn = self.model.model.layers[0].self_attn
        B, N = 1, 16
        hidden = torch.randn(
            B, N, self.model.config.hidden_size, dtype=torch.float16, device="cuda"
        )
        pos = torch.arange(N, device="cuda").unsqueeze(0)

        with torch.no_grad():
            cos, sin = self.model.model.rotary_emb(hidden, pos)
            for eps in [0.001, 0.01, 0.1, 1.0]:
                wrapper = Qwen3HybridGQABubbleWrapper(
                    original_attn=original_attn, epsilon=eps, lam=0.5,
                ).cuda()
                out, _ = wrapper(hidden, position_embeddings=(cos, sin))
                self.assertTrue(torch.isfinite(out).all(), f"NaN at eps={eps}")

    def test_wrapper_with_output_attentions_real(self):
        """output_attentions=True should return attention matrix."""
        from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

        original_attn = self.model.model.layers[0].self_attn
        B, N = 1, 16
        hidden = torch.randn(
            B, N, self.model.config.hidden_size, dtype=torch.float16, device="cuda"
        )
        pos = torch.arange(N, device="cuda").unsqueeze(0)

        with torch.no_grad():
            cos, sin = self.model.model.rotary_emb(hidden, pos)
            wrapper = Qwen3HybridGQABubbleWrapper(
                original_attn=original_attn, epsilon=0.01, lam=0.5,
            ).cuda()
            out, attn = wrapper(
                hidden, position_embeddings=(cos, sin), output_attentions=True
            )

        self.assertEqual(out.shape, (B, N, self.model.config.hidden_size))
        self.assertIsNotNone(attn)
        self.assertEqual(
            attn.shape, (B, self.model.config.num_attention_heads, N, N)
        )

    def test_swap_all_layers_real(self):
        """Should be able to swap all 28 layers and run a forward pass."""
        from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

        # Save original self_attn references for restoration.
        original_self_attns = [
            (i, layer.self_attn)
            for i, layer in enumerate(self.model.model.layers)
        ]

        try:
            for i, layer in enumerate(self.model.model.layers):
                original_attn = original_self_attns[i][1]
                wrapper = Qwen3HybridGQABubbleWrapper(
                    original_attn=original_attn, epsilon=0.01, lam=0.5,
                ).cuda()
                layer.self_attn = wrapper

            # Forward pass through full model
            B, N = 1, 8
            input_ids = torch.randint(0, 1000, (B, N), device="cuda")
            with torch.no_grad():
                out = self.model(input_ids).logits

            self.assertEqual(out.shape, (B, N, self.model.config.vocab_size))
            self.assertTrue(torch.isfinite(out).all())
        finally:
            # Restore original layers (so subsequent tests don't break).
            for i, orig_attn in original_self_attns:
                self.model.model.layers[i].self_attn = orig_attn


if __name__ == "__main__":
    unittest.main(verbosity=2)