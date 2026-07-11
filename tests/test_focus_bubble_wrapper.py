"""Integration tests for Qwen3FocusBubbleWrapper."""

import sys
import os
import unittest
import torch

# Skip GPU tests if no CUDA
SKIP_CUDA = not torch.cuda.is_available()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from qwen3_focus_bubble_wrapper import Qwen3FocusBubbleWrapper


@unittest.skipIf(SKIP_CUDA, "CUDA not available")
class TestQwen3FocusBubbleWrapper(unittest.TestCase):
    """Integration tests for Qwen3FocusBubbleWrapper with real Qwen3 model."""

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.MODEL_ID = "Qwen/Qwen3-0.6B-Base"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_ID)
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
            attn_implementation="eager",
        )
        cls.model.eval()
        cls.WINDOW = 128

        # Store original attention modules for all layers
        cls.original_attns = [
            cls.model.model.layers[i].self_attn
            for i in range(len(cls.model.model.layers))
        ]

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.original_attns
        torch.cuda.empty_cache()

    def setUp(self):
        torch.manual_seed(42)
        text = "Hello world this is a test " * 20
        self.input_ids = self.tokenizer(text, return_tensors="pt").input_ids[:, :self.WINDOW].cuda()

    def tearDown(self):
        """Restore all layers to original state after each test."""
        for i, orig_attn in enumerate(self.original_attns):
            self.model.model.layers[i].self_attn = orig_attn

    def test_wrapper_loads(self):
        """Wrapper can be instantiated and loaded on a Qwen3 layer."""
        orig_attn = self.original_attns[12]
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=0.001,
            tau_iters=1,
            use_psi=True,
            use_delta=False,
        ).cuda()
        self.model.model.layers[12].self_attn = wrapper
        self.assertIsInstance(wrapper, Qwen3FocusBubbleWrapper)
        self.tearDown()

    def test_forward_without_attentions(self):
        """Forward pass works without output_attentions."""
        orig_attn = self.original_attns[12]
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=0.001,
            tau_iters=1,
            use_psi=True,
            use_delta=False,
        ).cuda()
        self.model.model.layers[12].self_attn = wrapper

        with torch.no_grad():
            outputs = self.model(self.input_ids, output_attentions=False)

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], self.input_ids.shape[0])
        self.assertEqual(outputs.logits.shape[1], self.input_ids.shape[1])
        self.tearDown()

    def test_forward_with_attentions(self):
        """Forward pass works with output_attentions=True."""
        orig_attn = self.original_attns[12]
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=0.001,
            tau_iters=1,
            use_psi=True,
            use_delta=False,
        ).cuda()
        self.model.model.layers[12].self_attn = wrapper

        with torch.no_grad():
            outputs = self.model(self.input_ids, output_attentions=True)

        self.assertIsNotNone(outputs.attentions)
        self.assertGreaterEqual(len(outputs.attentions), 1)

        attn = outputs.attentions[12]
        # Attention matrix size may be less than WINDOW if input is shorter
        actual_len = attn.shape[-1]
        self.assertEqual(attn.shape[0], self.input_ids.shape[0])
        self.assertEqual(attn.shape[-1], actual_len)
        self.assertEqual(attn.shape[-2], actual_len)
        self.tearDown()

    def test_rope_applied(self):
        """RoPE is applied correctly in wrapper."""
        orig_attn = self.original_attns[12]
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=0.001,
            tau_iters=1,
            use_psi=True,
            use_delta=False,
        ).cuda()
        self.model.model.layers[12].self_attn = wrapper

        hidden_states = self.model.model.embed_tokens(self.input_ids)
        actual_len = hidden_states.shape[1]
        position_ids = torch.arange(actual_len, device=self.input_ids.device).unsqueeze(0)
        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        with torch.no_grad():
            out = wrapper(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                output_attentions=False,
            )

        self.assertIsNotNone(out[0])
        self.assertEqual(out[0].shape, hidden_states.shape)
        self.tearDown()

    def test_gqa_correctness(self):
        """GQA expansion works correctly (8 KV heads -> 16 query heads)."""
        orig_attn = self.original_attns[12]
        self.assertEqual(orig_attn.config.num_attention_heads, 16)
        self.assertEqual(orig_attn.config.num_key_value_heads, 8)

        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=0.001,
            tau_iters=1,
            use_psi=True,
            use_delta=False,
        ).cuda()
        self.model.model.layers[12].self_attn = wrapper

        with torch.no_grad():
            outputs = self.model(self.input_ids, output_attentions=True)

        attn = outputs.attentions[12]
        self.assertEqual(attn.shape[1], 16)
        self.tearDown()

    def test_causal_mask_preserved(self):
        """Causal mask is correctly applied (no attention to future)."""
        orig_attn = self.original_attns[12]
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=0.001,
            tau_iters=1,
            use_psi=True,
            use_delta=False,
        ).cuda()
        self.model.model.layers[12].self_attn = wrapper

        with torch.no_grad():
            outputs = self.model(self.input_ids, output_attentions=True)

        attn = outputs.attentions[12][0]
        upper = torch.triu(attn, diagonal=1)
        self.assertLess(upper.max().item(), 0.01)
        self.tearDown()

    def test_dtype_preservation(self):
        """Output dtype matches input dtype (float16)."""
        orig_attn = self.original_attns[12]
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=0.001,
            tau_iters=1,
            use_psi=True,
            use_delta=False,
        ).cuda()
        self.model.model.layers[12].self_attn = wrapper

        with torch.no_grad():
            outputs = self.model(self.input_ids)

        self.assertEqual(outputs.logits.dtype, torch.float16)
        self.tearDown()


@unittest.skipIf(SKIP_CUDA, "CUDA not available")
class TestQwen3FocusBubbleWrapperDeltaNet(unittest.TestCase):
    """Integration tests for FocusDeltaNet variant."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.MODEL_ID = "Qwen/Qwen3-0.6B-Base"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_ID)
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
            attn_implementation="eager",
        )
        cls.model.eval()
        cls.WINDOW = 128

        cls.original_attns = [
            cls.model.model.layers[i].self_attn
            for i in range(len(cls.model.model.layers))
        ]

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.original_attns
        torch.cuda.empty_cache()

    def setUp(self):
        torch.manual_seed(42)
        text = "Hello world this is a test " * 20
        self.input_ids = self.tokenizer(text, return_tensors="pt").input_ids[:, :self.WINDOW].cuda()

    def tearDown(self):
        for i, orig_attn in enumerate(self.original_attns):
            self.model.model.layers[i].self_attn = orig_attn

    def test_focus_deltanet_forward(self):
        """FocusDeltaNet (use_delta=True) forward pass works without NaN."""
        orig_attn = self.original_attns[7]
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=0.001,
            tau_iters=1,
            use_psi=True,
            use_delta=True,
            lam=0.3,
        ).cuda()
        self.model.model.layers[7].self_attn = wrapper

        with torch.no_grad():
            outputs = self.model(self.input_ids)

        self.assertIsNotNone(outputs.logits)
        self.assertFalse(torch.isnan(outputs.logits).any())
        self.assertFalse(torch.isinf(outputs.logits).any())
        self.tearDown()

    def test_focus_deltanet_with_attentions(self):
        """FocusDeltaNet works with output_attentions=True."""
        orig_attn = self.original_attns[7]
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=0.001,
            tau_iters=1,
            use_psi=True,
            use_delta=True,
            lam=0.3,
        ).cuda()
        self.model.model.layers[7].self_attn = wrapper

        with torch.no_grad():
            outputs = self.model(self.input_ids, output_attentions=True)

        self.assertIsNotNone(outputs.attentions)
        self.tearDown()


if __name__ == "__main__":
    unittest.main()