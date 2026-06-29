"""
Real vs Mock comparison: Hybrid vs Plateau on Qwen3-0.6B real embeddings
vs synthetic mock embeddings.

Validates whether findings from mock mode replicate on real embeddings.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, "experiments")
import numpy as np


def main():
    # Load both sweep results
    with open("results/epsilon_sweep_comparison.json") as f:
        mock_data = json.load(f)
    with open("results_real/real_sweep_comparison.json") as f:
        real_data = json.load(f)

    print("=" * 90)
    print("  HYBRID vs PLATEAU: MOCK vs REAL (Qwen3-0.6B) COMPARISON")
    print("=" * 90)

    # Mock baseline
    mock_baseline = mock_data["baseline"]
    print(f"\nMock baseline (synthetic embeddings):")
    print(f"  eff_rank: {mock_baseline['effective_rank']:.1f}")
    print(f"  intrinsic_dim: {mock_baseline['intrinsic_dim_mle']:.1f}")

    # Real baseline (layer 7)
    real_baseline = real_data["baseline_layer_7"]
    print(f"\nReal baseline (Qwen3-0.6B layer 7):")
    print(f"  eff_rank: {real_baseline['effective_rank']:.1f}")
    print(f"  intrinsic_dim: {real_baseline['intrinsic_dim_mle']:.1f}")
    print(f"  anisotropy: {real_baseline['anisotropy_index']:.3f}")

    # Hybrid results comparison
    mock_h = mock_data["hybrid_lam_0.5"]
    real_h = real_data["hybrid_lam_0.5"]
    mock_p = mock_data["plateau_legacy"]
    real_p = real_data["plateau_legacy"]

    print("\n" + "=" * 90)
    print(f"  {'eps':>6} | {'MOCK Hybrid':>14} | {'MOCK Plateau':>14} | {'REAL Hybrid':>14} | {'REAL Plateau':>14}")
    print(f"  {'':>6} | {'eff_rank':>8} {'rank_r':>5} | {'eff_rank':>8} {'rank_r':>5} | {'eff_rank':>8} {'rank_r':>5} | {'eff_rank':>8} {'rank_r':>5}")
    print("-" * 90)

    mock_base_er = mock_baseline["effective_rank"]
    real_base_er = real_baseline["effective_rank"]

    # Find max eps in mock (uses 9 values) and real (4 values)
    eps_mock = [r["eps"] for r in mock_h]
    eps_real = [r["eps"] for r in real_h]
    common_eps = sorted(set(eps_mock) & set(eps_real))

    for eps in common_eps:
        mh = next(r for r in mock_h if abs(r["eps"] - eps) < 1e-6)
        mp = next(r for r in mock_p if abs(r["eps"] - eps) < 1e-6)
        rh = next(r for r in real_h if abs(r["eps"] - eps) < 1e-6)
        rp = next(r for r in real_p if abs(r["eps"] - eps) < 1e-6)
        mh_rr = mh["effective_rank"] / mock_base_er
        mp_rr = mp["effective_rank"] / mock_base_er
        rh_rr = rh["effective_rank"] / real_base_er
        rp_rr = rp["effective_rank"] / real_base_er
        print(
            f"  {eps:>6.4f} | "
            f"{mh['effective_rank']:>8.1f} {mh_rr:>5.2f} | "
            f"{mp['effective_rank']:>8.1f} {mp_rr:>5.2f} | "
            f"{rh['effective_rank']:>8.1f} {rh_rr:>5.2f} | "
            f"{rp['effective_rank']:>8.1f} {rp_rr:>5.2f}"
        )

    print("\n" + "=" * 90)
    print("  CONCLUSIONS")
    print("=" * 90)

    # Average rank_ratio across common eps
    mh_avg = np.mean([next(r for r in mock_h if abs(r["eps"] - eps) < 1e-6)["effective_rank"] / mock_base_er for eps in common_eps])
    mp_avg = np.mean([next(r for r in mock_p if abs(r["eps"] - eps) < 1e-6)["effective_rank"] / mock_base_er for eps in common_eps])
    rh_avg = np.mean([next(r for r in real_h if abs(r["eps"] - eps) < 1e-6)["effective_rank"] / real_base_er for eps in common_eps])
    rp_avg = np.mean([next(r for r in real_p if abs(r["eps"] - eps) < 1e-6)["effective_rank"] / real_base_er for eps in common_eps])

    print(f"\n  Average rank_ratio (eff_rank / baseline_eff_rank):")
    print(f"    Mock:   Hybrid={mh_avg:.3f}, Plateau={mp_avg:.3f}, Hybrid advantage={mh_avg - mp_avg:+.3f}")
    print(f"    Real:   Hybrid={rh_avg:.3f}, Plateau={rp_avg:.3f}, Hybrid advantage={rh_avg - rp_avg:+.3f}")

    # Verdict
    h_better_mock = mh_avg > mp_avg
    h_better_real = rh_avg > rp_avg
    print(f"\n  Verdict:")
    print(f"    Mock:  Hybrid {'OUTPERFORMS' if h_better_mock else 'underperforms'} Plateau on rank_ratio")
    print(f"    Real:  Hybrid {'OUTPERFORMS' if h_better_real else 'underperforms'} Plateau on rank_ratio")

    if h_better_mock and h_better_real:
        print(f"\n  [VERDICT] Mock findings REPLICATE on real Qwen3-0.6B embeddings.")
        print(f"  HybridAttention (DeltaNet + SIRI + psi) is the recommended architecture.")
    elif h_better_mock != h_better_real:
        print(f"\n  [VERDICT] Mock findings DO NOT fully replicate on real embeddings.")
        print(f"  Further investigation needed.")
    else:
        print(f"\n  [VERDICT] Both modes favor Plateau. Hybrid may need tuning for real data.")

    # Save comparison
    comparison = {
        "mock_baseline": mock_baseline,
        "real_baseline_layer_7": real_baseline,
        "mock_hybrid_avg_rank_ratio": mh_avg,
        "mock_plateau_avg_rank_ratio": mp_avg,
        "real_hybrid_avg_rank_ratio": rh_avg,
        "real_plateau_avg_rank_ratio": rp_avg,
        "common_eps_values": common_eps,
        "verdict": {
            "hybrid_better_mock": h_better_mock,
            "hybrid_better_real": h_better_real,
            "mock_replicates_real": h_better_mock and h_better_real,
        },
    }
    Path("results_real").mkdir(exist_ok=True)
    with open("results_real/mock_vs_real_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nSaved comparison to results_real/mock_vs_real_comparison.json")


if __name__ == "__main__":
    main()