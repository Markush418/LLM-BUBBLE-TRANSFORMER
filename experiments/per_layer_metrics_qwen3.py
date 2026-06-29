"""Per-layer analysis of Qwen3-0.6B embeddings."""
import sys, json
from pathlib import Path
sys.path.insert(0, "experiments")
import numpy as np
from metrics import effective_rank, intrinsic_dimension_mle, anisotropy_index


def main():
    base_dir = Path("embeddings_all_28/softmax")
    layers = sorted([int(f.stem.split("_")[1]) for f in base_dir.glob("layer_*.npy")])

    results = []
    for layer_idx in layers:
        emb_path = base_dir / f"layer_{layer_idx}.npy"
        # Subsample for fast metrics computation (6400 tokens x 1024 is slow).
        emb = np.load(emb_path).astype(np.float32)
        flat = emb.reshape(-1, 1024)
        if flat.shape[0] > 1000:
            np.random.seed(42)
            idx = np.random.choice(flat.shape[0], 1000, replace=False)
            flat = flat[idx]
        er = effective_rank(flat)
        idim = intrinsic_dimension_mle(flat, k=10)
        ani = anisotropy_index(flat)
        results.append({"layer": layer_idx, "effective_rank": er,
                        "intrinsic_dim": idim, "anisotropy": ani})
        print(f"Layer {layer_idx:2d}: eff_rank={er:>7.1f}  intrinsic_dim={idim:>5.1f}  anisotropy={ani:.3f}")

    out_path = Path("results_real/per_layer_metrics_qwen3.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary statistics
    eff_ranks = [r["effective_rank"] for r in results]
    print(f"\nEffective rank summary (all 28 layers):")
    print(f"  min={min(eff_ranks):.1f}, max={max(eff_ranks):.1f}, mean={np.mean(eff_ranks):.1f}, median={np.median(eff_ranks):.1f}")

    # Identify low-rank layers (potential candidates for Hybrid replacement)
    low_rank_layers = [r for r in results if r["effective_rank"] < 50]
    print(f"\nLow-rank layers (eff_rank < 50): {[r['layer'] for r in low_rank_layers]}")
    print(f"Mid-rank layers (50 <= eff_rank < 200): {[r['layer'] for r in results if 50 <= r['effective_rank'] < 200]}")
    print(f"High-rank layers (eff_rank >= 200): {[r['layer'] for r in results if r['effective_rank'] >= 200]}")


if __name__ == "__main__":
    main()