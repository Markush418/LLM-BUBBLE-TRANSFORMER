"""
Real-mode micro-sweep for HybridAttention vs PlateauAttention.

Uses real Qwen3-0.6B embeddings but subsamples to 1 text of length 64
for fast sweep.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "experiments")
import numpy as np

from hybrid_attention import HybridAttention
from plateau_attention import PlateauAttentionMechanism
from metrics import compute_all_metrics


def load_raw_input(emb_dir: str) -> np.ndarray:
    path = Path(emb_dir) / "raw_input.npy"
    if not path.exists():
        return None
    raw = np.load(path).astype(np.float32)
    return raw  # [num_texts, max_length, d_model]


def main():
    emb_dir = "embeddings_real"
    eps_values = [0.001, 0.01, 0.1, 1.0]
    target_layers = [0, 7, 15, 23, 27]

    raw_input = load_raw_input(emb_dir)
    if raw_input is None:
        print(f"[ERROR] {emb_dir}/raw_input.npy not found.")
        sys.exit(1)

    # Use 1 sample, truncate to seq_len=64
    seq_len = 64
    raw_input_sub = raw_input[:1, :seq_len, :]  # [1, 64, 1024]
    print(f"raw_input_sub shape: {raw_input_sub.shape}, d_model: {raw_input_sub.shape[-1]}")

    # Compute baseline metrics from layer 7 (a middle layer)
    baseline_layer = np.load(f"{emb_dir}/softmax/layer_7.npy").astype(np.float32)
    baseline_flat = baseline_layer.reshape(-1, baseline_layer.shape[-1])
    base_metrics = compute_all_metrics(baseline_flat)
    print(f"\nBaseline (layer 7, raw Qwen3-0.6B):")
    print(f"  eff_rank: {base_metrics['effective_rank']:.1f}")
    print(f"  intrinsic_dim: {base_metrics['intrinsic_dim_mle']:.1f}")
    print(f"  anisotropy: {base_metrics['anisotropy_index']:.3f}")
    print()

    # Hybrid sweep
    print("=== HYBRID ATTENTION (DeltaNet + SIRI, lam=0.5) on REAL Qwen3-0.6B ===")
    print(f"{'eps':>8} | {'eff_rank':>10} | {'intr_dim':>10} | {'anisotropy':>10} | {'conc':>6} | {'rank_ratio':>10} | {'time':>6}")
    print("-" * 90)

    D = raw_input_sub.shape[-1]
    hybrid_results = []
    for eps in eps_values:
        attn = HybridAttention(d_model=D, num_heads=4, epsilon=eps, lam=0.5, chunk_size=8, seed=42)
        t0 = time.time()
        out, A = attn(raw_input_sub, return_attention=True)
        elapsed = time.time() - t0
        m = compute_all_metrics(out, A)
        rr = m["effective_rank"] / base_metrics["effective_rank"]
        print(
            f"{eps:>8.4f} | {m['effective_rank']:>10.1f} | "
            f"{m['intrinsic_dim_mle']:>10.1f} | "
            f"{m['anisotropy_index']:>10.3f} | "
            f"{m.get('concentration_ratio', 0):>6.3f} | "
            f"{rr:>10.2f} | {elapsed:>5.1f}s"
        )
        hybrid_results.append({"eps": eps, "lam": 0.5, "rank_ratio": rr, **m})

    # Plateau sweep
    print("\n=== PLATEAU ATTENTION (legacy SIRI) on REAL Qwen3-0.6B ===")
    print(f"{'eps':>8} | {'eff_rank':>10} | {'intr_dim':>10} | {'anisotropy':>10} | {'conc':>6} | {'rank_ratio':>10} | {'time':>6}")
    print("-" * 90)

    plateau_results = []
    for eps in eps_values:
        attn = PlateauAttentionMechanism(d_model=D, num_heads=4, epsilon=eps, tau_iters=5, seed=42)
        t0 = time.time()
        out, A = attn(raw_input_sub, return_attention=True)
        elapsed = time.time() - t0
        m = compute_all_metrics(out, A)
        rr = m["effective_rank"] / base_metrics["effective_rank"]
        print(
            f"{eps:>8.4f} | {m['effective_rank']:>10.1f} | "
            f"{m['intrinsic_dim_mle']:>10.1f} | "
            f"{m['anisotropy_index']:>10.3f} | "
            f"{m.get('concentration_ratio', 0):>6.3f} | "
            f"{rr:>10.2f} | {elapsed:>5.1f}s"
        )
        plateau_results.append({"eps": eps, "rank_ratio": rr, **m})

    # Sweet spots
    def best(results, base_r):
        best, best_s = None, -1
        for r in results:
            rr = r.get("rank_ratio", 0)
            conc = r.get("concentration_ratio", 0)
            if rr > 0.5:
                s = conc * (rr ** 0.5)
                if s > best_s:
                    best_s, best = s, r
        return best

    h_sweet = best(hybrid_results, base_metrics["effective_rank"])
    p_sweet = best(plateau_results, base_metrics["effective_rank"])

    print("\n=== SWEET SPOTS (real Qwen3-0.6B) ===")
    if h_sweet:
        print(f"  Hybrid (lam=0.5): eps={h_sweet['eps']:.4f}, eff_rank={h_sweet['effective_rank']:.1f}, "
              f"intr_dim={h_sweet['intrinsic_dim_mle']:.1f}, rank_ratio={h_sweet['rank_ratio']:.2f}")
    if p_sweet:
        print(f"  Plateau (legacy): eps={p_sweet['eps']:.4f}, eff_rank={p_sweet['effective_rank']:.1f}, "
              f"intr_dim={p_sweet['intrinsic_dim_mle']:.1f}, rank_ratio={p_sweet['rank_ratio']:.2f}")

    Path("results_real").mkdir(exist_ok=True)
    with open("results_real/real_sweep_comparison.json", "w") as f:
        json.dump({
            "mode": "real (Qwen3-0.6B)",
            "d_model": D,
            "n_samples": 1,
            "seq_len": seq_len,
            "baseline_layer_7": base_metrics,
            "hybrid_lam_0.5": hybrid_results,
            "plateau_legacy": plateau_results,
            "sweet_spot_hybrid": h_sweet,
            "sweet_spot_plateau": p_sweet,
        }, f, indent=2, default=str)
    print("\nSaved to results_real/real_sweep_comparison.json")


if __name__ == "__main__":
    main()