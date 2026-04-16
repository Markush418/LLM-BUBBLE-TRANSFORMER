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
COST_COLORS = {
    "l2_sq": "#FF6B6B",
    "cosine": "#4ECDC4",
    "dot_product": "#45B7D1",
    "mahalanobis": "#96CEB4",
    "mesh_learnable": "#FFEAA7",
}
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


def plot_cost_comparison_pareto(results: List[Dict]):
    """Scatter plot: concentration vs effective rank, colored by cost function."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Group by cost_type
    cost_groups = {}
    for r in results:
        ct = r.get("cost_type", "l2_sq")
        cost_groups.setdefault(ct, []).append(r)

    for cost_type, group in cost_groups.items():
        concentrations = [
            r.get("concentration_ratio", 0) for r in group if r.get("epsilon", 0) > 0
        ]
        eff_ranks = [
            r.get("effective_rank", 0) for r in group if r.get("epsilon", 0) > 0
        ]
        anisotropies = [
            r.get("anisotropy_index", 0.5) for r in group if r.get("epsilon", 0) > 0
        ]

        if not concentrations:
            continue

        color = COST_COLORS.get(cost_type, "gray")
        sizes = [30 + (1 - a) * 200 for a in anisotropies]  # Larger = lower anisotropy

        ax.scatter(
            concentrations,
            eff_ranks,
            c=color,
            s=sizes,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=cost_type,
        )

    # Pareto frontier region (bottom-right = best)
    ax.axvline(x=0.3, color="green", linestyle="--", alpha=0.3)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.3)

    ax.set_xlabel("Concentration Ratio (lower = more concentrated)", fontsize=13)
    ax.set_ylabel("Effective Rank (higher = more expressive)", fontsize=13)
    ax.set_title(
        "Cost Function Pareto Frontier — Plan E: Cost Matrix Engineering",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(
        loc="best",
        fontsize=11,
        framealpha=0.9,
        title="Cost Function",
        title_fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    _save_plot("cost_comparison_pareto")


def plot_tension_analysis(results_path: str = "results/tension_sweep.json"):
    """Generate tension sweep analysis plots (Plan C)."""
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"[Viz] Tension results not found: {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    baseline_ranks = data.get("baseline_ranks", {})
    optimal = data.get("optimal_alpha", {})

    if not results:
        return

    print(f"[Viz] Generating tension analysis plots ({len(results)} data points)...")

    # Group by layer
    layer_groups = {}
    for r in results:
        layer_groups.setdefault(r["layer"], []).append(r)

    # 1. Alpha vs Effective Rank (per layer)
    fig, ax = plt.subplots(figsize=(12, 7))
    for layer, group in sorted(layer_groups.items()):
        # Filtrar los que tienen 'alpha' (dual-head) antes de ordenar
        dual_group = [x for x in group if "alpha" in x]
        group = sorted(dual_group, key=lambda x: x["alpha"])
        alphas = [r["alpha"] for r in group if r.get("alpha", -1) >= 0]
        ranks = [r["effective_rank"] for r in group if r.get("alpha", -1) >= 0]
        if alphas:
            ax.plot(
                alphas, ranks, "o-", label=f"Layer {layer}", linewidth=2, markersize=6
            )
    ax.set_xlabel("Alpha (tension coefficient)")
    ax.set_ylabel("Effective Rank")
    ax.set_title(
        "Effective Rank vs Tension Coefficient", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=9)
    _save_plot("tension_alpha_vs_rank")

    # 2. Alpha vs Concentration Ratio (per layer)
    fig, ax = plt.subplots(figsize=(12, 7))
    for layer, group in sorted(layer_groups.items()):
        group = sorted(group, key=lambda x: x["alpha"])
        alphas = [r["alpha"] for r in group if r.get("alpha", -1) >= 0]
        concs = [
            r.get("concentration_ratio", 0) for r in group if r.get("alpha", -1) >= 0
        ]
        if alphas:
            ax.plot(
                alphas, concs, "o-", label=f"Layer {layer}", linewidth=2, markersize=6
            )
    ax.set_xlabel("Alpha (tension coefficient)")
    ax.set_ylabel("Concentration Ratio")
    ax.set_title("Concentration vs Tension Coefficient", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    _save_plot("tension_alpha_vs_concentration")

    # 3. Pareto: Concentration vs Rank (colored by layer)
    fig, ax = plt.subplots(figsize=(12, 8))
    for layer, group in sorted(layer_groups.items()):
        alphas = [r["alpha"] for r in group if r.get("alpha", -1) >= 0]
        concs = [
            r.get("concentration_ratio", 0) for r in group if r.get("alpha", -1) >= 0
        ]
        ranks = [r["effective_rank"] for r in group if r.get("alpha", -1) >= 0]
        if alphas:
            sc = ax.scatter(concs, ranks, s=80, alpha=0.7, label=f"Layer {layer}")
    # Optimal alpha marker
    if optimal.get("alpha") is not None:
        # Find corresponding point (approximate by nearest)
        opt_alpha = optimal["alpha"]
        closest = min(
            [r for r in results if r.get("alpha", -1) >= 0],
            key=lambda x: abs(x["alpha"] - opt_alpha),
        )
        ax.scatter(
            closest.get("concentration_ratio", 0),
            closest["effective_rank"],
            s=200,
            color="gold",
            edgecolors="black",
            linewidth=2,
            marker="*",
            label=f"Optimal α={opt_alpha}",
            zorder=10,
        )
    ax.set_xlabel("Concentration Ratio (lower = more concentrated)")
    ax.set_ylabel("Effective Rank (higher = more expressive)")
    ax.set_title("Tension Pareto Frontier", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    _save_plot("tension_pareto")

    # 4. Tension Balance Heatmap (layers x alpha)
    fig, ax = plt.subplots(figsize=(12, 7))
    alphas = sorted(set(r["alpha"] for r in results if r.get("alpha", -1) >= 0))
    layers = sorted(layer_groups.keys())
    matrix = np.zeros((len(layers), len(alphas)))
    for r in results:
        alpha = r.get("alpha", -1)
        if alpha < 0:
            continue
        l_idx = layers.index(r["layer"])
        a_idx = alphas.index(alpha)
        matrix[l_idx, a_idx] = r.get("tension_balance", 0)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", origin="lower")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Layer")
    ax.set_title("Tension Balance Heatmap", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Tension Balance (0=identical, 1=orthogonal)")
    _save_plot("tension_balance_heatmap")

    print("[Viz] Tension analysis plots generated!")


def plot_layer_selection(results_path: str = "results/layer_selection.json"):
    """Generate layer selection comparison plots."""
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"[Viz] Layer selection results not found: {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    # Convert JSON string keys back to integers for layer indices
    baseline = {int(k): v for k, v in data.get("baseline", {}).items()}
    plateau = {int(k): v for k, v in data.get("plateau", {}).items()}
    ranked_layers = [int(l) for l in data.get("ranked_layers", [])]
    layer_scores = {int(k): v for k, v in data.get("layer_scores", {}).items()}

    if not baseline or not plateau or not ranked_layers:
        print("[Viz] Insufficient data for layer selection plots")
        return

    layers = sorted([l for l in ranked_layers if l in baseline and l in plateau])
    if not layers:
        return

    print(f"[Viz] Generating layer selection plots for {len(layers)} layers...")

    # 1. Bar chart: Effective Rank (baseline vs plateau)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(layers))
    width = 0.35
    baseline_ranks = [baseline[l]["effective_rank"] for l in layers]
    plateau_ranks = [plateau[l]["effective_rank"] for l in layers]
    ax.bar(
        x - width / 2,
        baseline_ranks,
        width,
        label="Softmax (baseline)",
        color="#4ECDC4",
    )
    ax.bar(
        x + width / 2, plateau_ranks, width, label="Plateau (ε=0.001)", color="#FF6B6B"
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Effective Rank")
    ax.set_title(
        "Effective Rank: Baseline vs PlateauAttention", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend()
    _save_plot("layer_selection_rank_comparison")

    # 2. Bar chart: Concentration Improvement (%)
    fig, ax = plt.subplots(figsize=(12, 7))
    improvements = []
    for l in layers:
        b_conc = baseline[l].get("concentration_ratio", 0)
        p_conc = plateau[l].get("concentration_ratio", 0)
        imp = (p_conc - b_conc) / b_conc * 100 if b_conc > 0 else 0
        improvements.append(imp)
    colors = ["green" if imp > 0 else "red" for imp in improvements]
    ax.bar(x, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Concentration Improvement (%)")
    ax.set_title(
        "Concentration Gain per Layer (Plateau vs Softmax)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    for i, imp in enumerate(improvements):
        ax.text(
            x[i],
            imp + (0.5 if imp >= 0 else -0.5),
            f"{imp:.1f}%",
            ha="center",
            va="bottom" if imp >= 0 else "top",
            fontsize=8,
        )
    _save_plot("layer_selection_concentration_gain")

    # 3. Bar chart: Intrinsic Dim Preservation (%)
    fig, ax = plt.subplots(figsize=(12, 7))
    preservation = []
    for l in layers:
        b_dim = baseline[l]["intrinsic_dim_mle"]
        p_dim = plateau[l]["intrinsic_dim_mle"]
        pres = (p_dim / b_dim) * 100 if b_dim > 0 else 0
        preservation.append(pres)
    ax.bar(x, preservation, color="#96CEB4", alpha=0.8)
    ax.axhline(y=100, color="red", linestyle="--", label="Baseline (100%)")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Intrinsic Dim Preservation (%)")
    ax.set_title(
        "Intrinsic Dimensionality Preservation", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend()
    for i, pres in enumerate(preservation):
        ax.text(x[i], pres + 0.5, f"{pres:.1f}%", ha="center", va="bottom", fontsize=8)
    _save_plot("layer_selection_intrinsic_dim_preservation")

    # 4. Scatter: Concentration vs Effective Rank (Pareto per layer)
    fig, ax = plt.subplots(figsize=(12, 8))
    for l in layers:
        b_conc = baseline[l].get("concentration_ratio", 0)
        b_rank = baseline[l]["effective_rank"]
        p_conc = plateau[l].get("concentration_ratio", 0)
        p_rank = plateau[l]["effective_rank"]
        # Plot baseline point
        ax.scatter(
            b_conc,
            b_rank,
            color="blue",
            s=100,
            alpha=0.6,
            marker="s",
            label="Baseline" if l == layers[0] else None,
        )
        # Plot plateau point
        ax.scatter(
            p_conc,
            p_rank,
            color="red",
            s=100,
            alpha=0.6,
            marker="o",
            label="Plateau" if l == layers[0] else None,
        )
        # Connect with arrow
        ax.annotate(
            "",
            xy=(p_conc, p_rank),
            xytext=(b_conc, b_rank),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )
        # Label layer
        mid_x = (b_conc + p_conc) / 2
        mid_y = (b_rank + p_rank) / 2
        ax.text(
            mid_x,
            mid_y,
            f"L{l}",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )
    ax.set_xlabel("Concentration Ratio (lower = more concentrated)")
    ax.set_ylabel("Effective Rank (higher = more expressive)")
    ax.set_title("Pareto Movement: Layer-wise Changes", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_plot("layer_selection_pareto_movement")

    # 5. Bar chart: Overall layer score (ranking)
    fig, ax = plt.subplots(figsize=(12, 7))
    scores = [layer_scores[l]["score"] for l in ranked_layers]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ranked_layers)))
    ax.bar(x, scores, color=colors, alpha=0.8)
    ax.set_xlabel("Layer Index (Sorted by Score)")
    ax.set_ylabel("Composite Score (higher = better)")
    ax.set_title("Layer Ranking by Improvement Score", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in ranked_layers])
    for i, s in enumerate(scores):
        ax.text(x[i], s + 0.01, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
    _save_plot("layer_selection_ranking")

    print("[Viz] Layer selection plots generated!")


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
    plot_cost_comparison_pareto(results)
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
