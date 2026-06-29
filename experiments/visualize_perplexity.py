"""Visualize perplexity benchmark results."""
import sys, json
from pathlib import Path
sys.path.insert(0, "experiments")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_final_benchmark(json_path: str, out_dir: str):
    with open(json_path) as f:
        results = json.load(f)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart: PPL per config
    keys = list(results.keys())
    ppls = [results[k]["ppl"] if results[k]["ppl"] == results[k]["ppl"] else 0 for k in keys]
    base = results["baseline"]["ppl"]
    deltas = [p - base if p > 0 else 0 for p in ppls]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PPL
    ax = axes[0]
    colors = ["#2ca02c" if k == "baseline" else "#1f77b4" for k in keys]
    bars = ax.bar(range(len(keys)), ppls, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Final Perplexity Benchmark: Hybrid vs Baseline\n(Qwen3-0.6B on WikiText-2 test, 50k chars)")
    ax.axhline(base, ls="--", color="gray", alpha=0.5, label=f"Baseline = {base:.2f}")
    for i, (k, p) in enumerate(zip(keys, ppls)):
        ax.text(i, p + 5, f"{p:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(ppls) * 1.15 if max(ppls) > 0 else 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "perplexity_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'perplexity_bar.png'}")

    # Delta PPL
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if k == "baseline" else "#d62728" if d > 5 else "#ff7f0e" if d > 0 else "#1f77b4" for k, d in zip(keys, deltas)]
    bars = ax.bar(range(len(keys)), deltas, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("ΔPPL vs Baseline")
    ax.set_title("Delta Perplexity (lower is better)\nPositive = worse than baseline")
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    for i, d in enumerate(deltas):
        ax.text(i, d + (1 if d >= 0 else -3), f"{d:+.2f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "perplexity_delta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'perplexity_delta.png'}")


def plot_lambda_sweep(json_path: str, out_dir: str):
    with open(json_path) as f:
        results = json.load(f)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lambda sweep on mid-layers
    sweep = [(k, v) for k, v in results.items() if "mid_L03-L15_lam" in k]
    sweep.sort(key=lambda x: x[1]["lambda"])
    lambdas = [v[1]["lambda"] for v in sweep]
    ppls = [v[1]["ppl"] for v in sweep]

    base = results["baseline"]["ppl"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lambdas, ppls, "o-", linewidth=2, markersize=10, color="#1f77b4",
            label="Hybrid (L03-L15)")
    ax.axhline(base, ls="--", color="green", alpha=0.7, label=f"Baseline ({base:.2f})")
    for x, y in zip(lambdas, ppls):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    ax.set_xlabel("λ (Hybrid interpolation weight)\nλ=0: pure SIRI, λ=1: pure DeltaNet")
    ax.set_ylabel("Perplexity")
    ax.set_title("Lambda Sweep on Qwen3-0.6B (Mid-layers L03-L15)\nWikiText-2 test, 50k chars")
    ax.set_xticks(lambdas)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_sweep_ppl.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'lambda_sweep_ppl.png'}")


def plot_layerwise(json_path: str, out_dir: str):
    with open(json_path) as f:
        results = json.load(f)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = results["baseline"]["ppl"]
    layers, ppls, deltas = [], [], []
    for k, v in results.items():
        if k.startswith("L") and v.get("lambda") == 1.0:
            layers.append(v["layer"])
            ppls.append(v["ppl"] if v["ppl"] == v["ppl"] else 0)
            deltas.append(v["delta_ppl"])

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#d62728" if d > 100 else "#ff7f0e" if d > 2 else "#2ca02c" for d in deltas]
    bars = ax.bar(layers, deltas, color=colors, width=0.8)
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("Layer index (Qwen3-0.6B)")
    ax.set_ylabel("ΔPPL vs Baseline")
    ax.set_title("Per-Layer Sensitivity: swap one layer with pure DeltaNet (λ=1.0)\nQwen3-0.6B on WikiText-2 test (20k chars)")
    ax.set_xticks(layers)
    ax.grid(axis="y", alpha=0.3)
    for layer, ppl, d in zip(layers, ppls, deltas):
        if ppl < 100:
            ax.text(layer, d + 3, f"{ppl:.1f}", ha="center", fontsize=8)
        else:
            ax.text(layer, min(d * 0.5, 100), f"{ppl:.0f}", ha="center", fontsize=8, color="darkred")
    fig.tight_layout()
    fig.savefig(out_dir / "per_layer_dPPL.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'per_layer_dPPL.png'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_real/perplexity_final")
    parser.add_argument("--layerwise-results", default="results_real/perplexity_layerwise/ppl_per_layer.json")
    parser.add_argument("--out-dir", default="results_real/perplexity_final/plots")
    args = parser.parse_args()

    plot_final_benchmark(Path(args.results_dir) / "ppl_final.json", args.out_dir)
    plot_lambda_sweep(Path(args.results_dir) / "ppl_final.json", args.out_dir)
    if Path(args.layerwise_results).exists():
        plot_layerwise(args.layerwise_results, args.out_dir)