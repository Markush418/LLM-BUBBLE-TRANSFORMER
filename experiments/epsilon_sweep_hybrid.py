"""Full epsilon sweep for HybridAttention: identify the sweet spot."""

import sys
import json
from pathlib import Path

sys.path.insert(0, "experiments")
import numpy as np

from hybrid_attention import HybridAttention
from plateau_attention import PlateauAttentionMechanism
from metrics import compute_all_metrics
from epsilon_sweep import load_raw_input, load_embeddings


def main():
    raw_input = load_raw_input("embeddings")
    d_model = raw_input.shape[-1]
    print(f"raw_input shape: {raw_input.shape}, d_model: {d_model}")

    baseline = load_embeddings("embeddings", 7)
    baseline_metrics = compute_all_metrics(baseline)
    print(f"\nBaseline (softmax): eff_rank={baseline_metrics['effective_rank']:.1f}")

    eps_values = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

    # Hybrid sweep at lam=0.5
    print(f"\n=== HYBRID ATTENTION (DeltaNet + SIRI, lam=0.5) ===")
    print(f"{'eps':>8} | {'eff_rank':>10} | {'intr_dim':>10} | {'anisotropy':>10} | {'conc':>6} | {'entropy':>7} | {'rank_ratio':>10}")
    print("-" * 90)

    hybrid_results = []
    for eps in eps_values:
        attn = HybridAttention(d_model=d_model, num_heads=4, epsilon=eps, lam=0.5, seed=42)
        out, A = attn(raw_input, return_attention=True)
        m = compute_all_metrics(
            out, A,
            baseline_effective_rank=baseline_metrics["effective_rank"],
        )
        print(
            f"{eps:>8.4f} | {m['effective_rank']:>10.1f} | "
            f"{m['intrinsic_dim_mle']:>10.1f} | "
            f"{m['anisotropy_index']:>10.3f} | "
            f"{m.get('concentration_ratio', 0):>6.3f} | "
            f"{m.get('attention_entropy', 0):>7.2f} | "
            f"{m.get('effective_rank_ratio', 0):>10.2f}"
        )
        hybrid_results.append({"eps": eps, "lam": 0.5, **m})

    # Plateau sweep (legacy SIRI) for comparison
    print(f"\n=== PLATEAU ATTENTION (legacy SIRI only) ===")
    print(f"{'eps':>8} | {'eff_rank':>10} | {'intr_dim':>10} | {'anisotropy':>10} | {'conc':>6} | {'entropy':>7} | {'rank_ratio':>10}")
    print("-" * 90)

    plateau_results = []
    for eps in eps_values:
        attn = PlateauAttentionMechanism(d_model=d_model, num_heads=4, epsilon=eps, tau_iters=5, seed=42)
        out, A = attn(raw_input, return_attention=True)
        m = compute_all_metrics(
            out, A,
            baseline_effective_rank=baseline_metrics["effective_rank"],
        )
        print(
            f"{eps:>8.4f} | {m['effective_rank']:>10.1f} | "
            f"{m['intrinsic_dim_mle']:>10.1f} | "
            f"{m['anisotropy_index']:>10.3f} | "
            f"{m.get('concentration_ratio', 0):>6.3f} | "
            f"{m.get('attention_entropy', 0):>7.2f} | "
            f"{m.get('effective_rank_ratio', 0):>10.2f}"
        )
        plateau_results.append({"eps": eps, **m})

    # Identify sweet spot for hybrid
    sweet_spot = None
    best_score = -1.0
    for r in hybrid_results:
        rank_ratio = r.get("effective_rank_ratio", 0)
        conc = r.get("concentration_ratio", 0)
        if rank_ratio > 0.5:
            score = conc * (rank_ratio ** 0.5)
            if score > best_score:
                best_score = score
                sweet_spot = r

    if sweet_spot:
        print(f"\n*** SWEET SPOT (Hybrid lam=0.5) ***")
        print(f"  eps:        {sweet_spot['eps']:.4f}")
        print(f"  eff_rank:   {sweet_spot['effective_rank']:.1f}")
        print(f"  intrinsic:  {sweet_spot['intrinsic_dim_mle']:.1f}")
        print(f"  anisotropy: {sweet_spot['anisotropy_index']:.3f}")
        print(f"  score:      {best_score:.4f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/epsilon_sweep_comparison.json", "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "hybrid_lam_0.5": hybrid_results,
            "plateau_legacy": plateau_results,
            "sweet_spot_hybrid": sweet_spot,
        }, f, indent=2, default=str)
    print("\nSaved to results/epsilon_sweep_comparison.json")


if __name__ == "__main__":
    main()