"""Lambda sweep for HybridAttention: explore DeltaNet vs SIRI balance."""

import sys
import json
from pathlib import Path

sys.path.insert(0, "experiments")
import numpy as np

from hybrid_attention import HybridAttention
from metrics import compute_all_metrics
from epsilon_sweep import load_raw_input, load_embeddings


def main():
    raw_input = load_raw_input("embeddings")
    d_model = raw_input.shape[-1]
    print(f"raw_input shape: {raw_input.shape}, d_model: {d_model}")

    # Baseline (softmax) metrics from layer 7
    baseline = load_embeddings("embeddings", 7)
    baseline_metrics = compute_all_metrics(baseline)
    print(f"\nBaseline (softmax, layer 7):")
    print(f"  eff_rank: {baseline_metrics['effective_rank']:.1f}")
    print(f"  intrinsic_dim: {baseline_metrics['intrinsic_dim_mle']:.1f}")
    print(f"  anisotropy: {baseline_metrics['anisotropy_index']:.3f}")
    print(f"  concentration: {baseline_metrics.get('concentration_ratio', 0):.3f}")
    print()

    print(f"{'lam':>6} | {'eff_rank':>10} | {'intr_dim':>10} | {'anisotropy':>10} | {'conc':>6} | {'entropy':>7} | {'eff_rank_ratio':>14}")
    print("-" * 90)

    results = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        attn = HybridAttention(d_model=d_model, num_heads=4, epsilon=0.1, lam=lam, seed=42)
        out, A = attn(raw_input, return_attention=True)
        m = compute_all_metrics(
            out, A,
            baseline_effective_rank=baseline_metrics["effective_rank"],
        )
        print(
            f"{lam:>6.2f} | {m['effective_rank']:>10.1f} | "
            f"{m['intrinsic_dim_mle']:>10.1f} | "
            f"{m['anisotropy_index']:>10.3f} | "
            f"{m.get('concentration_ratio', 0):>6.3f} | "
            f"{m.get('attention_entropy', 0):>7.2f} | "
            f"{m.get('effective_rank_ratio', 0):>14.2f}"
        )
        results.append({
            "lam": lam,
            "epsilon": 0.1,
            "baseline_eff_rank": baseline_metrics["effective_rank"],
            **m,
        })

    Path("results").mkdir(exist_ok=True)
    with open("results/lambda_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nSaved to results/lambda_sweep_results.json")
    print(f"\nInterpretation:")
    print(f"  lam=0.0  -> Pure SIRI (Doubly-stochastic via Sinkhorn)")
    print(f"  lam=0.25 -> Mostly SIRI")
    print(f"  lam=0.5  -> Balanced hybrid")
    print(f"  lam=0.75 -> Mostly DeltaNet")
    print(f"  lam=1.0  -> Pure DeltaNet (linear attention, no SIRI)")


if __name__ == "__main__":
    main()