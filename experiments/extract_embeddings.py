"""
Embedding Extractor — Qwen3-0.6B Real Embeddings (4-bit Quantized)
====================================================================
Loads Qwen3-0.6B-Base in 4-bit quantization (bitsandbytes) for
GTX 1650 (4GB VRAM) + 8GB RAM systems.

Extracts hidden states from every transformer layer via forward hooks.
Saves embeddings as .npy files compatible with the existing pipeline.

Usage:
    python extract_embeddings.py
    python extract_embeddings.py --model "Qwen/Qwen3-0.6B-Base" --batch-size 2
    python extract_embeddings.py --skip-download  # use cached model

Hardware targets:
    - GPU: GTX 1650 (4GB VRAM)
    - RAM: 8GB
    - Quantization: 4-bit NF4 (bitsandbytes)
    - Memory footprint: ~600MB VRAM + ~1.5GB RAM
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm


# ─── Constants ──────────────────────────────────────────────────────────────

# Qwen3-0.6B-Base architecture (from config.json)
QWEN3_06B_CONFIG = {
    "d_model": 1024,
    "num_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
}

# Default model — 0.6B fits in 4GB VRAM with 4-bit quantization
DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"

# All layers are full attention in Qwen3-0.6B (no DeltaNet like in larger models)
ALL_LAYERS = list(range(28))


# ─── Embedding Extractor ────────────────────────────────────────────────────


class QwenEmbeddingExtractor:
    """
    Loads Qwen3 in 4-bit quantization and extracts hidden states per layer.

    Uses bitsandbytes NF4 quantization to fit in 4GB VRAM.
    Forward hooks capture hidden states AFTER each transformer layer
    (post-residual, which is what the attention mechanism outputs into).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        batch_size: int = 2,
        max_length: int = 512,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.layer_outputs: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

        print(f"[Extractor] Loading {model_name} in 4-bit NF4...")
        print(
            f"[Extractor] Device: {device}, Batch: {batch_size}, Max len: {max_length}"
        )

        self._load_model(trust_remote_code)
        print(f"[Extractor] Model loaded. {self._get_memory_usage()}")

    def _load_model(self, trust_remote_code: bool = True):
        """Load model with 4-bit quantization for low-VRAM systems."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import BitsAndBytesConfig

        # 4-bit NF4 quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # extra compression: ~0.4 bits/token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=self.device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,
        )
        self.model.eval()

        # Register hooks on every transformer layer
        # Qwen3 uses model.model.layers[i] for transformer blocks
        layers = self.model.model.layers
        print(f"[Extractor] Found {len(layers)} transformer layers")

        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(
                lambda mod, inp, out, idx=i: self._capture(idx, out)
            )
            self.hooks.append(hook)

    def _capture(self, layer_idx: int, output):
        """Forward hook: capture hidden state from layer output."""
        # Qwen3 layer output is a tuple: (hidden_states,) or (hidden_states, past_key_values, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # Move to CPU immediately to save VRAM
        # Convert to float32 to prevent overflow from quantization artifacts
        self.layer_outputs[layer_idx] = hidden_states.float().cpu()

    def _get_memory_usage(self) -> str:
        """Return current GPU memory usage string."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            return f"VRAM: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved"
        return "CPU only"

    def tokenize_corpus(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a list of texts into input_ids."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

    def extract(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[int, np.ndarray]:
        """
        Run forward pass and return hidden states per layer.

        Returns:
            {layer_idx: np.ndarray of shape [B, N, D]}
        """
        self.layer_outputs.clear()

        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Convert to numpy (float32 for numerical stability)
        result = {}
        for layer_idx, hidden in self.layer_outputs.items():
            # hidden: [B, N, D] — already on CPU from hook, float32
            result[layer_idx] = hidden.numpy().astype(np.float32)

        return result

    def extract_corpus(self, texts: List[str]) -> Dict[int, np.ndarray]:
        """
        Extract embeddings from a full corpus in batches.

        Returns:
            {layer_idx: np.ndarray of shape [total_tokens, D]}
            where total_tokens = sum of sequence lengths across all texts
        """
        # Accumulate per-layer embeddings across batches
        layer_accum: Dict[int, List[np.ndarray]] = {
            i: [] for i in range(len(self.model.model.layers))
        }

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting batches"):
            batch_texts = texts[i : i + self.batch_size]
            tokenized = self.tokenize_corpus(batch_texts)

            layer_outputs = self.extract(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )

            for layer_idx, hidden in layer_outputs.items():
                # Flatten batch dimension: [B, N, D] -> [B*N, D]
                flat = hidden.reshape(-1, hidden.shape[-1])
                layer_accum[layer_idx].append(flat)

            # Clear VRAM between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all batches per layer
        result = {}
        for layer_idx in layer_accum:
            if layer_accum[layer_idx]:
                result[layer_idx] = np.concatenate(layer_accum[layer_idx], axis=0)
            else:
                result[layer_idx] = np.array([])

        return result

    def cleanup(self):
        """Remove hooks and free memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── Corpus Loading ─────────────────────────────────────────────────────────


def load_corpus(corpus_path: str) -> List[str]:
    """Load texts from a JSONL file."""
    texts = []
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue

    print(f"[Corpus] Loaded {len(texts)} texts from {corpus_path}")
    return texts


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract real embeddings from Qwen3-0.6B (4-bit quantized)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/test_corpus.jsonl",
        help="Path to JSONL corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="embeddings",
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size (reduce for low VRAM)"
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="Only extract these layers (default: all 28)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Qwen3-0.6B Embedding Extractor — 4-bit NF4 Quantized")
    print("  LLM-BUBBLE — Real Embeddings Mode")
    print("=" * 70)
    print()

    start_time = time.time()

    # ─── Step 1: Load Corpus ────────────────────────────────────────────
    print("[Step 1/3] Loading corpus...")
    texts = load_corpus(args.corpus)
    if not texts:
        print("[ERROR] No texts found in corpus!")
        sys.exit(1)
    print()

    # ─── Step 2: Load Model + Extract ───────────────────────────────────
    print("[Step 2/3] Loading model and extracting embeddings...")
    print("-" * 50)

    try:
        extractor = QwenEmbeddingExtractor(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        print("[Hint] Make sure bitsandbytes is installed:")
        print("  pip install bitsandbytes accelerate transformers")
        print("[Hint] On Windows, you may need:")
        print("  pip install bitsandbytes-windows")
        sys.exit(1)

    layer_embeddings = extractor.extract_corpus(texts)

    # ─── Step 3: Save Embeddings ────────────────────────────────────────
    print("\n[Step 3/3] Saving embeddings...")
    print("-" * 50)

    output_dir = Path(args.output_dir)
    softmax_dir = output_dir / "softmax"
    softmax_dir.mkdir(parents=True, exist_ok=True)

    target_layers = args.target_layers or sorted(layer_embeddings.keys())

    saved_count = 0
    for layer_idx in target_layers:
        if layer_idx not in layer_embeddings:
            print(f"  Layer {layer_idx}: SKIPPED (no data)")
            continue

        emb = layer_embeddings[layer_idx]
        save_path = softmax_dir / f"layer_{layer_idx}.npy"
        np.save(save_path, emb)

        # Quick effective rank check
        emb_flat = emb.astype(np.float32)
        centered = emb_flat - emb_flat.mean(axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        S = S[S > 1e-10]
        if len(S) > 0:
            p = S / S.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            eff_rank = float(np.exp(entropy))
        else:
            eff_rank = 0.0

        print(f"  Layer {layer_idx:2d}: shape={emb.shape}, eff_rank={eff_rank:.1f}")
        saved_count += 1

    # Save raw input (layer 0 pre-attention = token embeddings)
    if 0 in layer_embeddings:
        raw_path = output_dir / "raw_input.npy"
        np.save(raw_path, layer_embeddings[0])
        print(f"  Raw input: {layer_embeddings[0].shape}")

    # Save metadata
    metadata = {
        "mode": "real",
        "model": args.model,
        "quantization": "4bit-NF4-double",
        "num_texts": len(texts),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": args.device,
        "layers_saved": saved_count,
        "d_model": QWEN3_06B_CONFIG["d_model"],
        "num_layers": QWEN3_06B_CONFIG["num_layers"],
        "num_attention_heads": QWEN3_06B_CONFIG["num_attention_heads"],
        "description": "Real embeddings from Qwen3-0.6B-Base (4-bit quantized)",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    extractor.cleanup()

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"  EXTRACTION COMPLETE — {elapsed:.1f}s")
    print("=" * 70)
    print(f"\n  Texts processed: {len(texts)}")
    print(f"  Layers saved: {saved_count}")
    print(f"  Output: {output_dir}/softmax/")
    print(f"  Metadata: {output_dir}/metadata.json")
    print()


if __name__ == "__main__":
    main()
