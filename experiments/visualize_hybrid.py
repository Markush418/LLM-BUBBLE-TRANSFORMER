"""
Visualization for HybridAttention vs Plateau comparison.
=========================================================

Generates 5 publication-quality plots comparing Hybrid (DeltaNet + SIRI + psi)
against Plateau (legacy SIRI) on both mock and real Qwen3-0.6B embeddings.

Outputs (saved to plots/):
  1. hybrid_vs_plateau_effective_rank.png    - eff_rank curves across eps
  2. hybrid_vs_plateau_concentration.png     - concentration across eps
  3. hybrid_vs_plateau_pareto.png            - rank vs concentration scatter
  4. mock_vs_real_verification.png           - mock vs real rank_ratio
  5. lambda_sweep_real.png                   - lambda sweep on real embeddings

Usage:
    python experiments/visualize_hybrid.py
    python experiments/visualize_hybrid.py --mock-results results/epsilon_sweep_comparison.json --real-results results_real/real_sweep_comparison.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid", font_scale=1.0)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use("ggplot")


# Colors
HYBRID_COLOR = "#2E86AB"     # blue
PLATEAU_COLOR = "#E63946"    # red
BASELINE_COLOR = "#888888"   # gray
MOCK_COLOR = "#06A77D"       # green
REAL_COLOR = "#F77F00"       # orange

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _save(name: str, fig, dpi: int = 150):
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved: {path}")
    return path


def _load_json(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"[Viz] WARN: {path} not found, skipping dependent plots.")
        return None
    with open(p, "r") as f:
        return json.load(f)


def plot_eps_sweep_comparison(
    mock_data: Optional[Dict] = None,
    real_data: Optional[Dict] = None,
):
    """Plot 1: Effective rank vs epsilon (Hybrid vs Plateau, mock + real)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mock
    ax = axes[0]
    if mock_data:
        mock_base = mock_data["baseline"]["effective_rank"]
        ax.axhline(mock_base, color=BASELINE_COLOR, linestyle="--",
                   linewidth=1.5, label=f"Softmax baseline ({mock_base:.0f})")
        for label, key, color in [("Hybrid (lam=0.5)", "hybrid_lam_0.5", HYBRID_COLOR),
                                   ("Plateau (legacy)", "plateau_legacy", PLATEAU_COLOR)]:
            data = mock_data.get(key, [])
            if data:
                eps = [r["eps"] for r in data]
                er = [r["effective_rank"] for r in data]
                ax.plot(eps, er, "o-", color=color, linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (bandwidth)")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Mock embeddings (synthetic)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Real
    ax = axes[1]
    if real_data:
        real_base = real_data["baseline_layer_7"]["effective_rank"]
        ax.axhline(real_base, color=BASELINE_COLOR, linestyle="--",
                   linewidth=1.5, label=f"Layer 7 baseline ({real_base:.0f})")
        for label, key, color in [("Hybrid (lam=0.5)", "hybrid_lam_0.5", HYBRID_COLOR),
                                   ("Plateau (legacy)", "plateau_legacy", PLATEAU_COLOR)]:
            data = real_data.get(key, [])
            if data:
                eps = [r["eps"] for r in data]
                er = [r["effective_rank"] for r in data]
                ax.plot(eps, er, "o-", color=color, linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (bandwidth)")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Real Qwen3-0.6B embeddings (layer 7)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Effective Rank vs Epsilon: Hybrid (DeltaNet+SIRI+psi) vs Plateau",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save("hybrid_vs_plateau_effective_rank", fig)


def plot_concentration_comparison(
    mock_data: Optional[Dict] = None,
    real_data: Optional[Dict] = None,
):
    """Plot 2: Concentration ratio vs epsilon (mock + real)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, key_baseline in [
        (axes[0], mock_data, "Mock embeddings (synthetic)", "baseline"),
        (axes[1], real_data, "Real Qwen3-0.6B (layer 7)", "baseline_layer_7"),
    ]:
        if not data:
            ax.set_title(f"{title} (data not available)")
            continue
        base = data.get(key_baseline, {})
        base_conc = base.get("concentration_ratio", 0)
        ax.axhline(base_conc, color=BASELINE_COLOR, linestyle="--",
                   linewidth=1.5, label=f"Baseline ({base_conc:.3f})")
        for label, key, color in [("Hybrid (lam=0.5)", "hybrid_lam_0.5", HYBRID_COLOR),
                                   ("Plateau (legacy)", "plateau_legacy", PLATEAU_COLOR)]:
            data_arr = data.get(key, [])
            if data_arr:
                eps = [r["eps"] for r in data_arr]
                conc = [r.get("concentration_ratio", 0) for r in data_arr]
                ax.plot(eps, conc, "o-", color=color, linewidth=2, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Epsilon (bandwidth)")
        ax.set_ylabel("Concentration Ratio")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Concentration Ratio vs Epsilon: Hybrid vs Plateau",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save("hybrid_vs_plateau_concentration", fig)


def plot_pareto(mock_data: Optional[Dict] = None, real_data: Optional[Dict] = None):
    """Plot 3: Pareto frontier - concentration vs eff_rank."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for label, data, key, color, marker in [
        ("Hybrid (mock)", mock_data, "hybrid_lam_0.5", HYBRID_COLOR, "o"),
        ("Plateau (mock)", mock_data, "plateau_legacy", PLATEAU_COLOR, "s"),
        ("Hybrid (real)", real_data, "hybrid_lam_0.5", MOCK_COLOR, "^"),
        ("Plateau (real)", real_data, "plateau_legacy", REAL_COLOR, "v"),
    ]:
        if not data:
            continue
        arr = data.get(key, [])
        if not arr:
            continue
        er = [r["effective_rank"] for r in arr]
        conc = [r.get("concentration_ratio", 0) for r in arr]
        ax.scatter(er, conc, c=color, marker=marker, s=80, label=label, alpha=0.8)

    ax.set_xlabel("Effective Rank")
    ax.set_ylabel("Concentration Ratio")
    ax.set_title("Pareto Frontier: Hybrid vs Plateau (mock + real)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save("hybrid_vs_plateau_pareto", fig)


def plot_mock_vs_real_verification(comparison: Optional[Dict] = None):
    """Plot 4: Mock vs real rank_ratio verification."""
    if not comparison:
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Mock", "Real"]
    hybrid_rr = [comparison.get("mock_hybrid_avg_rank_ratio", 0),
                 comparison.get("real_hybrid_avg_rank_ratio", 0)]
    plateau_rr = [comparison.get("mock_plateau_avg_rank_ratio", 0),
                   comparison.get("real_plateau_avg_rank_ratio", 0)]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, hybrid_rr, width, label="Hybrid",
                  color=HYBRID_COLOR, edgecolor="black")
    bars2 = ax.bar(x + width/2, plateau_rr, width, label="Plateau",
                  color=PLATEAU_COLOR, edgecolor="black")

    # Annotate
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10)

    ax.axhline(1.0, color="gray", linestyle="--", label="baseline (=1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Average rank_ratio (eff_rank / baseline_eff_rank)")
    ax.set_title("Hybrid vs Plateau: Mock findings REPLICATE on real Qwen3-0.6B")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Verdict text
    verdict = comparison.get("verdict", {})
    if verdict.get("mock_replicates_real"):
        text = "[VERIFIED] Mock findings replicate on real embeddings."
        color = "green"
    else:
        text = "[WARNING] Mock and real findings differ."
        color = "red"
    ax.text(0.5, 0.02, text, transform=ax.transAxes, ha="center",
            fontsize=11, fontweight="bold", color=color,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    _save("mock_vs_real_verification", fig)


def plot_lambda_sweep_real(lam_data: Optional[Dict] = None):
    """Plot 5: Lambda sweep on real embeddings."""
    if not lam_data:
        return
    results = lam_data.get("lambda_sweep_results", [])
    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lam = [r["lam"] for r in results]
    er = [r.get("effective_rank", r.get("eff_rank", 0)) for r in results]
    rr = [r["rank_ratio"] for r in results]

    # Panel 1: eff_rank vs lambda
    axes[0].plot(lam, er, "o-", color=HYBRID_COLOR, linewidth=2, markersize=10)
    axes[0].axvline(0.0, color=PLATEAU_COLOR, linestyle="--",
                    alpha=0.6, label="pure SIRI (lam=0)")
    axes[0].axvline(1.0, color=HYBRID_COLOR, linestyle="--",
                    alpha=0.6, label="pure DeltaNet (lam=1)")
    axes[0].set_xlabel("Lambda (DeltaNet<->SIRI)")
    axes[0].set_ylabel("Effective Rank")
    axes[0].set_title("Effective Rank vs Lambda (real Qwen3-0.6B)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: rank_ratio vs lambda
    axes[1].plot(lam, rr, "o-", color=HYBRID_COLOR, linewidth=2, markersize=10)
    axes[1].axhline(1.0, color=BASELINE_COLOR, linestyle="--",
                    linewidth=1.5, label="baseline (=1.0)")
    axes[1].set_xlabel("Lambda (DeltaNet<->SIRI)")
    axes[1].set_ylabel("rank_ratio (eff_rank / baseline)")
    axes[1].set_title("Rank Ratio vs Lambda (real Qwen3-0.6B)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Lambda Sweep on Real Qwen3-0.6B Embeddings",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save("lambda_sweep_real", fig)


def generate_all_plots(
    mock_results_path: str = "results/epsilon_sweep_comparison.json",
    real_results_path: str = "results_real/real_sweep_comparison.json",
    mock_vs_real_path: str = "results_real/mock_vs_real_comparison.json",
    lambda_real_path: str = "results_real/lambda_sweep_real.json",
):
    """Generate all 5 plots from the data."""
    print("[Viz] Loading data...")
    mock_data = _load_json(mock_results_path)
    real_data = _load_json(real_results_path)
    comparison = _load_json(mock_vs_real_path)
    lam_data = _load_json(lambda_real_path)

    print("[Viz] Generating plots...")
    if mock_data or real_data:
        plot_eps_sweep_comparison(mock_data, real_data)
        plot_concentration_comparison(mock_data, real_data)
        plot_pareto(mock_data, real_data)

    if comparison:
        plot_mock_vs_real_verification(comparison)

    if lam_data:
        plot_lambda_sweep_real(lam_data)

    print("[Viz] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize HybridAttention vs Plateau")
    parser.add_argument("--mock-results", type=str,
                        default="results/epsilon_sweep_comparison.json")
    parser.add_argument("--real-results", type=str,
                        default="results_real/real_sweep_comparison.json")
    parser.add_argument("--mock-vs-real", type=str,
                        default="results_real/mock_vs_real_comparison.json")
    parser.add_argument("--lambda-real", type=str,
                        default="results_real/lambda_sweep_real.json")
    args = parser.parse_args()
    generate_all_plots(
        mock_results_path=args.mock_results,
        real_results_path=args.real_results,
        mock_vs_real_path=args.mock_vs_real,
        lambda_real_path=args.lambda_real,
    )
