"""
Synthetic Embedding Generator — Mock Mode (NumPy Only)
========================================================
Generates realistic synthetic embeddings that mimic Qwen 3.6's
embedding geometry without requiring the actual model.

Key properties preserved:
  - Per-layer evolution of effective rank
  - Anisotropy patterns across layers
  - Pairwise distance distributions

Usage:
    python generate_mock_embeddings.py
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def generate_layer_embeddings(
    num_layers: int = 24,
    batch_size: int = 4,
    seq_len: int = 64,
    d_model: int = 512,
    num_heads: int = 8,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic embeddings mimicking Qwen 3.6 per-layer evolution.
    Early layers: high rank, isotropic. Late layers: lower rank, anisotropic.
    """
    rng = np.random.RandomState(seed)
    embeddings = {}

    for layer_idx in range(num_layers):
        layer_progress = layer_idx / max(num_layers - 1, 1)
        target_rank = int(d_model * (0.8 - 0.5 * layer_progress))
        target_rank = max(target_rank, 10)
        anisotropy = 0.02 + 0.15 * layer_progress

        total_tokens = batch_size * seq_len
        U = rng.randn(total_tokens, target_rank).astype(np.float32)
        V = rng.randn(target_rank, d_model).astype(np.float32)

        scales = np.ones(d_model, dtype=np.float32)
        dominant_dims = int(d_model * anisotropy)
        scales[:dominant_dims] = 3.0 + 2.0 * rng.rand(dominant_dims)

        emb = U @ V
        emb = emb * scales[np.newaxis, :]
        noise_level = 0.1 * (1 - layer_progress * 0.5)
        emb = emb + noise_level * rng.randn(*emb.shape).astype(np.float32)

        norms = np.linalg.norm(emb, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb = emb / norms
        emb = emb.reshape(batch_size, seq_len, d_model)
        embeddings[layer_idx] = emb

    return embeddings


def generate_raw_input_embeddings(
    batch_size: int = 4, seq_len: int = 64, d_model: int = 512, seed: int = 42
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vocab_size = 10000
    vocab_embeddings = rng.randn(vocab_size, d_model).astype(np.float32)
    vocab_embeddings = vocab_embeddings / np.linalg.norm(
        vocab_embeddings, axis=-1, keepdims=True
    )
    token_ids = rng.randint(0, vocab_size, size=(batch_size, seq_len))
    return vocab_embeddings[token_ids]


def generate_attention_mask(
    batch_size: int = 4, seq_len: int = 64, seed: int = 42
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mask = np.ones((batch_size, seq_len), dtype=np.int64)
    for b in range(batch_size):
        if rng.random() < 0.1:
            pad_start = rng.randint(seq_len // 2, seq_len)
            mask[b, pad_start:] = 0
    return mask


def _quick_effective_rank(emb: np.ndarray) -> float:
    emb_flat = emb.reshape(-1, emb.shape[-1]).astype(np.float32)
    centered = emb_flat - emb_flat.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(entropy))


def save_mock_embeddings(
    output_dir: str = "embeddings",
    num_layers: int = 24,
    batch_size: int = 4,
    seq_len: int = 64,
    d_model: int = 512,
    num_heads: int = 8,
    seed: int = 42,
):
    output_dir = Path(output_dir)
    softmax_dir = output_dir / "softmax"
    softmax_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Mock] Generating synthetic embeddings...")
    print(f"  Layers: {num_layers}, Batch: {batch_size} x {seq_len} x {d_model}")

    layer_embeddings = generate_layer_embeddings(
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        seed=seed,
    )

    for layer_idx, emb in layer_embeddings.items():
        save_path = softmax_dir / f"layer_{layer_idx}.npy"
        np.save(save_path, emb)
        eff_rank = _quick_effective_rank(emb)
        print(f"  Layer {layer_idx:2d}: shape={emb.shape}, eff_rank={eff_rank:.1f}")

    raw_input = generate_raw_input_embeddings(
        batch_size=batch_size, seq_len=seq_len, d_model=d_model, seed=seed
    )
    np.save(output_dir / "raw_input.npy", raw_input)
    print(f"  Raw input: {raw_input.shape}")

    attention_mask = generate_attention_mask(
        batch_size=batch_size, seq_len=seq_len, seed=seed
    )
    np.save(output_dir / "attention_mask.npy", attention_mask)

    metadata = {
        "mode": "mock",
        "num_layers": num_layers,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_heads": num_heads,
        "seed": seed,
        "description": "Synthetic embeddings mimicking Qwen 3.6 geometry (numpy-only)",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Mock] Saved to {output_dir}/")
    print(f"[Mock] {len(layer_embeddings)} layers + raw_input + metadata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mock embeddings (numpy-only)"
    )
    parser.add_argument("--output-dir", type=str, default="embeddings")
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_mock_embeddings(
        output_dir=args.output_dir,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        seed=args.seed,
    )
