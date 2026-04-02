"""
Visualization Engine — NumPy Implementation
==============================================
Generates publication-quality plots for the Plan A+B experiment.
Pure NumPy input — no PyTorch required.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_theme(style="darkgrid", font="sans-serif", font_scale=1.1)

COLORS = sns.color_palette("husl", 10)
EPSILON_COLORS = dict(
    zip(
        [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        sns.color_palette("viridis", 9),
    )
)
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _save_plot(name: str, dpi: int = 200):
    path = PLOT_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {path}")
    return path


def plot_effective_rank_curves(
    results: List[Dict], baseline_ranks: Optional[Dict] = None
):
    fig, ax = plt.subplots(figsize=(12, 7))
    epsilon_groups = {}
    for r in results:
        eps = r.get("epsilon", 0.0)
        epsilon_groups.setdefault(eps, []).append(r)

    for eps in sorted(epsilon_groups.keys()):
        group = epsilon_groups[eps]
        layers = [r["layer"] for r in group]
        ranks = [r["effective_rank"] for r in group]
        sorted_pairs = sorted(zip(layers, ranks))
        layers, ranks = zip(*sorted_pairs)
        color = EPSILON_COLORS.get(eps, "gray")
        label = f"eps = {eps}" if eps > 0 else "Softmax (baseline)"
        ax.plot(
            layers, ranks, "o-", color=color, label=label, linewidth=2, markersize=6
        )

    if baseline_ranks:
        bl_layers = sorted(baseline_ranks.keys())
        bl_ranks = [baseline_ranks[l] for l in bl_layers]
        ax.plot(
            bl_layers,
            bl_ranks,
            "s--",
            color="red",
            label="Softmax baseline",
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Effective Rank", fontsize=13)
    ax.set_title(
        "Effective Rank vs Layer — Sinkhorn vs Softmax", fontsize=15, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    _save_plot("effective_rank_curves")


def plot_concentration_heatmap(
    results: List[Dict], metric: str = "concentration_ratio"
):
    epsilons = sorted(
        set(r.get("epsilon", 0.0) for r in results if r.get("epsilon", 0.0) > 0)
    )
    layers = sorted(set(r["layer"] for r in results))
    matrix = np.zeros((len(layers), len(epsilons)))

    for r in results:
        eps = r.get("epsilon", 0.0)
        if eps <= 0:
            continue
        l_idx = layers.index(r["layer"])
        e_idx = epsilons.index(eps)
        matrix[l_idx, e_idx] = r.get(metric, 0.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", origin="lower")
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"{e:.3f}" for e in epsilons], rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Epsilon", fontsize=13)
    ax.set_ylabel("Layer Index", fontsize=13)
    ax.set_title(f"Concentration Heatmap: {metric}", fontsize=15, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, fontsize=12)

    for i in range(len(layers)):
        for j in range(len(epsilons)):
            val = matrix[i, j]
            text_color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    plt.tight_layout()
    _save_plot(f"concentration_heatmap_{metric}")


def plot_pareto_frontier(results: List[Dict]):
    fig, ax = plt.subplots(figsize=(10, 8))
    for r in results:
        eps = r.get("epsilon", 0.0)
        cr = r.get("concentration_ratio", 0.0)
        er = r.get("effective_rank", 0.0)
        layer = r.get("layer", 0)
        if eps <= 0 or cr <= 0:
            continue
        color = EPSILON_COLORS.get(eps, "gray")
        ax.scatter(
            cr,
            er,
            color=color,
            s=80 + layer * 5,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.annotate(
            f"L{layer}",
            (cr, er),
            fontsize=7,
            alpha=0.8,
            textcoords="offset points",
            xytext=(5, 3),
        )

    ax.axvline(x=0.3, color="green", linestyle="--", alpha=0.5)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5)
    ax.fill_betweenx([0, 600], 0, 0.3, alpha=0.1, color="green")
    ax.set_xlabel("Concentration Ratio (lower = more concentrated)", fontsize=12)
    ax.set_ylabel("Effective Rank (higher = more expressive)", fontsize=12)
    ax.set_title(
        "Pareto Frontier: Concentration vs Expressivity", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_plot("pareto_frontier")


def plot_anisotropy_vs_epsilon(results: List[Dict]):
    fig, ax = plt.subplots(figsize=(12, 7))
    layer_groups = {}
    for r in results:
        layer_groups.setdefault(r["layer"], []).append(r)

    for layer in sorted(layer_groups.keys()):
        group = layer_groups[layer]
        epsilons = [r["epsilon"] for r in group if r["epsilon"] > 0]
        anisotropy = [r["anisotropy_index"] for r in group if r["epsilon"] > 0]
        sorted_pairs = sorted(zip(epsilons, anisotropy))
        if not sorted_pairs:
            continue
        epsilons, anisotropy = zip(*sorted_pairs)
        ax.plot(
            epsilons,
            anisotropy,
            "o-",
            label=f"Layer {layer}",
            linewidth=2,
            markersize=6,
        )

    ax.axhline(
        y=0.5, color="red", linestyle="--", alpha=0.5, label="Collapse threshold"
    )
    ax.set_xlabel("Epsilon", fontsize=13)
    ax.set_ylabel("Anisotropy Index", fontsize=13)
    ax.set_title("Anisotropy vs Epsilon", fontsize=15, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    _save_plot("anisotropy_vs_epsilon")


def plot_intrinsic_dim_vs_epsilon(results: List[Dict]):
    fig, ax = plt.subplots(figsize=(12, 7))
    layer_groups = {}
    for r in results:
        layer_groups.setdefault(r["layer"], []).append(r)

    for layer in sorted(layer_groups.keys()):
        group = layer_groups[layer]
        epsilons = [r["epsilon"] for r in group if r["epsilon"] > 0]
        dims = [r["intrinsic_dim_mle"] for r in group if r["epsilon"] > 0]
        sorted_pairs = sorted(zip(epsilons, dims))
        if not sorted_pairs:
            continue
        epsilons, dims = zip(*sorted_pairs)
        ax.plot(epsilons, dims, "o-", label=f"Layer {layer}", linewidth=2, markersize=6)

    ax.set_xlabel("Epsilon", fontsize=13)
    ax.set_ylabel("Intrinsic Dimensionality (MLE)", fontsize=13)
    ax.set_title("Intrinsic Dimensionality vs Epsilon", fontsize=15, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    _save_plot("intrinsic_dim_vs_epsilon")


def plot_tsne_embeddings(
    embeddings_dict: Dict[int, np.ndarray], layer_indices: Optional[List[int]] = None
):
    from sklearn.manifold import TSNE

    if layer_indices is None:
        layer_indices = sorted(embeddings_dict.keys())[:3]

    for layer_idx in layer_indices:
        if layer_idx not in embeddings_dict:
            continue
        emb = embeddings_dict[layer_idx]
        if emb.ndim == 3:
            emb = emb.reshape(-1, emb.shape[-1])
        if emb.shape[0] > 500:
            indices = np.random.choice(emb.shape[0], 500, replace=False)
            emb = emb[indices]

        print(f"[Viz] Computing t-SNE for layer {layer_idx} ({emb.shape})...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, emb.shape[0] - 1),
            random_state=42,
            init="pca",
        )
        emb_2d = tsne.fit_transform(emb)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            emb_2d[:, 0],
            emb_2d[:, 1],
            c=range(len(emb_2d)),
            cmap="viridis",
            alpha=0.7,
            s=30,
        )
        ax.set_title(f"t-SNE: Layer {layer_idx}", fontsize=15, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        plt.colorbar(scatter, ax=ax, label="Token Index")
        plt.tight_layout()
        _save_plot(f"tsne_layer_{layer_idx}")


def generate_summary_dashboard(results: List[Dict], sweet_spot: Dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epsilons = sorted(
        set(r.get("epsilon", 0.0) for r in results if r.get("epsilon", 0.0) > 0)
    )
    layers = sorted(set(r["layer"] for r in results))

    ax = axes[0, 0]
    for eps in epsilons[:5]:
        group = [r for r in results if r.get("epsilon") == eps]
        ls = [r["layer"] for r in group]
        rs = [r["effective_rank"] for r in group]
        sorted_pairs = sorted(zip(ls, rs))
        if sorted_pairs:
            ls, rs = zip(*sorted_pairs)
            ax.plot(ls, rs, "o-", label=f"eps={eps}", linewidth=1.5, markersize=4)
    ax.set_title("Effective Rank by Layer", fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Eff. Rank")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    matrix = np.zeros((len(layers), len(epsilons)))
    for r in results:
        eps = r.get("epsilon", 0.0)
        if eps <= 0:
            continue
        l_idx = layers.index(r["layer"])
        e_idx = epsilons.index(eps)
        matrix[l_idx, e_idx] = r.get("concentration_ratio", 0.0)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", origin="lower")
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"{e:.2f}" for e in epsilons], rotation=45, fontsize=7)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=7)
    ax.set_title("Concentration Ratio", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    for layer in layers[:4]:
        group = [r for r in results if r["layer"] == layer and r.get("epsilon", 0) > 0]
        es = [r["epsilon"] for r in group]
        ans = [r["anisotropy_index"] for r in group]
        sorted_pairs = sorted(zip(es, ans))
        if sorted_pairs:
            es, ans = zip(*sorted_pairs)
            ax.plot(es, ans, "o-", label=f"L{layer}", linewidth=1.5, markersize=4)
    ax.set_xscale("log")
    ax.set_title("Anisotropy vs eps", fontsize=12, fontweight="bold")
    ax.set_xlabel("eps")
    ax.set_ylabel("Anisotropy")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.axis("off")
    text = f"""SWEET SPOT ANALYSIS
{"=" * 30}

Recommended eps: {sweet_spot.get("epsilon", "N/A")}
Recommended layers: {sweet_spot.get("layers", "N/A")}

Expected concentration: {sweet_spot.get("concentration_ratio", "N/A"):.4f}
Expected effective rank: {sweet_spot.get("effective_rank", "N/A"):.1f}
Expected intrinsic dim: {sweet_spot.get("intrinsic_dim_mle", "N/A"):.1f}

Confidence: {sweet_spot.get("confidence", "N/A")}
"""
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    plt.suptitle(
        "Bubble Transformer — Embedding Geometry Analysis Summary",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save_plot("summary_dashboard")


def generate_all_plots(results_path: str = "results/epsilon_sweep.json"):
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"[Viz] Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    baseline_ranks = data.get("baseline_ranks", {})
    sweet_spot = data.get("sweet_spot", {})
    print(f"[Viz] Loaded {len(results)} results")

    plot_effective_rank_curves(results, baseline_ranks)
    plot_concentration_heatmap(results, metric="concentration_ratio")
    plot_concentration_heatmap(results, metric="effective_rank")
    plot_pareto_frontier(results)
    plot_anisotropy_vs_epsilon(results)
    plot_intrinsic_dim_vs_epsilon(results)
    generate_summary_dashboard(results, sweet_spot)
    print("[Viz] All plots generated!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualization plots")
    parser.add_argument("--results", type=str, default="results/epsilon_sweep.json")
    args = parser.parse_args()
    generate_all_plots(args.results)
