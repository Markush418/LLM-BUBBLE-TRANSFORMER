"""
Embedding Extractor Simplified (no bitsandbytes)
==================================================

GTX 1650-friendly Qwen3-0.6B extractor that runs in bfloat16 without
4-bit quantization. Qwen3-0.6B in bfloat16 = ~1.2GB VRAM; with overhead
fits comfortably in 4GB VRAM.

Pipeline:
  1. Load Qwen3-0.6B-Base in bfloat16 (no quantization).
  2. Run forward pass on a small corpus (batch_size=1, max_length=256).
  3. Capture hidden states from every transformer layer via forward hooks.
  4. Save per-layer embeddings [B, N, D] to embeddings/softmax/layer_*.npy.
  5. Save raw_input (token embeddings pre-layer-0) to embeddings/raw_input.npy.
  6. Write metadata.json.

Compatibility:
  - Output format matches run_hybrid_experiment.py and epsilon_sweep.py.
  - Saved shape: [B, N, D] per layer (NOT flattened as in the bitsandbytes version).

Usage:
    python experiments/extract_embeddings_simple.py
    python experiments/extract_embeddings_simple.py --batch-size 1 --max-length 256
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


# Qwen3-0.6B-Base architecture
QWEN3_06B_CONFIG = {
    "d_model": 1024,
    "num_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "intermediate_size": 3072,
}
DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
ALL_LAYERS = list(range(28))


class QwenEmbeddingExtractorSimple:
    """Qwen3-0.6B extractor in bfloat16 (no quantization).

    Designed for GTX 1650 (4GB VRAM). Uses bfloat16 weights + bfloat16
    forward pass to fit in memory.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.layer_outputs: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

        print(f"[Extractor-Simple] Loading {model_name} in bfloat16...")
        print(f"[Extractor-Simple] Device: {device}, Max len: {max_length}")

        self._load_model()
        print(f"[Extractor-Simple] Model loaded. {self._get_memory_usage()}")

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()

        layers = self.model.model.layers
        print(f"[Extractor-Simple] Found {len(layers)} transformer layers")

        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(
                lambda mod, inp, out, idx=i: self._capture(idx, out)
            )
            self.hooks.append(hook)

    def _capture(self, layer_idx: int, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # Keep on GPU as bfloat16 to save VRAM; convert to float32 CPU at save time.
        self.layer_outputs[layer_idx] = hidden_states.detach()

    def _get_memory_usage(self) -> str:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            return f"VRAM: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved"
        return "CPU only"

    def extract_batch(
        self, texts: List[str]
    ) -> Dict[int, np.ndarray]:
        """Extract embeddings from a single batch.

        Returns:
            {layer_idx: np.ndarray of shape [B, N, D]}
        """
        self.layer_outputs.clear()
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            self.model(**inputs)

        result = {}
        for layer_idx, hidden in self.layer_outputs.items():
            # Move to CPU + cast to float32 for stability
            result[layer_idx] = hidden.cpu().float().numpy().astype(np.float32)
        return result

    def extract_corpus(self, texts: List[str]) -> Dict[int, np.ndarray]:
        """Process texts one at a time (batch_size=1) for memory safety.

        Pads each text to exactly max_length so all outputs have the same
        sequence dimension and can be concatenated.

        Returns:
            {layer_idx: np.ndarray of shape [total_texts, max_length, D]}
        """
        layer_accum: Dict[int, List[np.ndarray]] = {
            i: [] for i in range(len(self.model.model.layers))
        }

        for i in tqdm(range(len(texts)), desc="Extracting"):
            text = texts[i : i + 1]  # batch_size = 1
            layer_outputs = self.extract_batch(text)
            for layer_idx, hidden in layer_outputs.items():
                # Pad to max_length on sequence axis (axis=1)
                if hidden.shape[1] < self.max_length:
                    pad_width = ((0, 0), (0, self.max_length - hidden.shape[1]), (0, 0))
                    hidden = np.pad(hidden, pad_width, mode="constant", constant_values=0)
                elif hidden.shape[1] > self.max_length:
                    hidden = hidden[:, : self.max_length, :]
                layer_accum[layer_idx].append(hidden)  # [1, max_length, D]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = {}
        for layer_idx in layer_accum:
            if layer_accum[layer_idx]:
                result[layer_idx] = np.concatenate(layer_accum[layer_idx], axis=0)
            else:
                result[layer_idx] = np.array([])
        return result

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def main():
    parser = argparse.ArgumentParser(
        description="Extract real Qwen3-0.6B embeddings (bfloat16, no quantization)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--corpus", type=str, default="data/test_corpus.jsonl")
    parser.add_argument("--output-dir", type=str, default="embeddings_real")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  Qwen3-0.6B Extractor (Simplified, bfloat16, GTX 1650)")
    print("=" * 70)
    print()
    start_time = time.time()

    print("[Step 1/3] Loading corpus...")
    texts = load_corpus(args.corpus)
    if not texts:
        print("[ERROR] No texts found in corpus!")
        sys.exit(1)
    print()

    print(f"[Step 2/3] Loading {args.model} and extracting...")
    print("-" * 50)
    try:
        extractor = QwenEmbeddingExtractorSimple(
            model_name=args.model,
            device=args.device,
            max_length=args.max_length,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        print("[Hint] Make sure transformers + torch are installed:")
        print("  pip install transformers accelerate torch")
        sys.exit(1)

    layer_embeddings = extractor.extract_corpus(texts)

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
        emb_flat = emb.reshape(-1, emb.shape[-1]).astype(np.float32)
        centered = emb_flat - emb_flat.mean(axis=0, keepdims=True)
        try:
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            S = S[S > 1e-10]
            if len(S) > 0:
                p = S / S.sum()
                entropy = -np.sum(p * np.log(p + 1e-10))
                eff_rank = float(np.exp(entropy))
            else:
                eff_rank = 0.0
        except (np.linalg.LinAlgError, ValueError):
            eff_rank = -1.0

        print(f"  Layer {layer_idx:2d}: shape={emb.shape}, eff_rank={eff_rank:.1f}")
        saved_count += 1

    # Save raw_input (layer 0 embeddings as proxy for token embeddings)
    if 0 in layer_embeddings:
        raw_path = output_dir / "raw_input.npy"
        # raw_input should be [B, N, D] - same as layer outputs
        np.save(raw_path, layer_embeddings[0])
        print(f"  Raw input: {layer_embeddings[0].shape}")

    metadata = {
        "mode": "real",
        "model": args.model,
        "quantization": "bfloat16 (no quantization)",
        "num_texts": len(texts),
        "max_length": args.max_length,
        "device": args.device,
        "layers_saved": saved_count,
        "d_model": QWEN3_06B_CONFIG["d_model"],
        "num_layers": QWEN3_06B_CONFIG["num_layers"],
        "num_attention_heads": QWEN3_06B_CONFIG["num_attention_heads"],
        "description": "Real embeddings from Qwen3-0.6B-Base (bfloat16)",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    extractor.cleanup()

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"  EXTRACTION COMPLETE \u2014 {elapsed:.1f}s")
    print("=" * 70)
    print(f"\n  Texts processed: {len(texts)}")
    print(f"  Layers saved: {saved_count}")
    print(f"  Output: {output_dir}/softmax/")
    print()


if __name__ == "__main__":
    main()