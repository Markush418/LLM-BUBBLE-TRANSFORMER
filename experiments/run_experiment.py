"""
Main Experiment Orchestrator — Plan A+B Combined
==================================================
Single entry point to run the full embedding geometry analysis.

Supports two modes:
  - MOCK: Synthetic embeddings (NumPy only, no GPU needed)
  - REAL: Real embeddings from Qwen3-0.6B (4-bit quantized, needs GPU)

Auto-detects mode from embeddings/metadata.json.
Override with --mode mock or --mode real.

Pipeline:
  1. Load or generate embeddings (real or mock)
  2. Run epsilon sweep across all target layers
  3. Compute metrics and generate visualizations
  4. Produce sweet spot analysis report

Usage:
    python run_experiment.py                    # auto-detect mode
    python run_experiment.py --mode mock        # force mock mode
    python run_experiment.py --mode real        # force real mode
    python run_experiment.py --d-model 256      # mock mode override
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def detect_mode(embeddings_dir: str) -> str:
    """Auto-detect whether we have real or mock embeddings."""
    metadata_path = Path(embeddings_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        return meta.get("mode", "mock")
    # Check if softmax embeddings exist at all
    softmax_dir = Path(embeddings_dir) / "softmax"
    if softmax_dir.exists() and any(softmax_dir.glob("layer_*.npy")):
        return "mock"  # default assumption if no metadata
    return "none"  # no embeddings at all


def main():
    parser = argparse.ArgumentParser(
        description="Plan A+B: Embedding Geometry + Epsilon Sweet Spot"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "mock", "real", "tension", "layer-selection"],
        help="Experiment mode: auto (detect), mock, real, tension (dual-head sweep), layer-selection",
    )
    parser.add_argument(
        "--d-model", type=int, default=None, help="Model hidden dimension (mock only)"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Number of attention heads (mock only)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=24,
        help="Number of layers to simulate (mock only)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--seq-len", type=int, default=64, help="Sequence length (mock only)"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="embeddings",
        help="Directory for embeddings",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory for results"
    )
    parser.add_argument(
        "--skip-generation", action="store_true", help="Skip embedding generation"
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--epsilon-values",
        type=float,
        nargs="+",
        default=None,
        help="Custom epsilon values",
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="Custom target layers",
    )
    parser.add_argument(
        "--cost-type",
        type=str,
        default="l2_sq",
        choices=[
            "l2_sq",
            "cosine",
            "dot_product",
            "mahalanobis",
            "mesh_learnable",
            "all",
        ],
        help="Cost function type for PlateauAttention (default: l2_sq)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # V4 Bubble Transformer arguments
    parser.add_argument(
        "--v4",
        action="store_true",
        help="Enable V4 (FPS + Expert-Choice routing)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=32,
        help="Number of experts/centroids for V4 routing (default: 32)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Tokens per expert in Expert-Choice routing (default: 8)",
    )
    parser.add_argument(
        "--no-fps-init",
        action="store_true",
        help="Disable FPS initialization (use random centroids)",
    )
    # GOAT arguments
    parser.add_argument(
        "--goat",
        action="store_true",
        help="Enable GOAT (Gated Optimal Attention Transport)",
    )
    parser.add_argument(
        "--no-learn-gates", action="store_true", help="Disable gate learning in GOAT"
    )
    parser.add_argument(
        "--gate-lr", type=float, default=0.01, help="GOAT gate learning rate"
    )
    # New metric flags
    parser.add_argument(
        "--spectral-metrics",
        action="store_true",
        default=True,
        help="Enable spectral metrics (SIGMA paper)",
    )
    parser.add_argument(
        "--crowding-metrics",
        action="store_true",
        default=True,
        help="Enable crowding metrics",
    )
    # Output format
    parser.add_argument(
        "--json-output", action="store_true", help="Output results as JSON"
    )
    args = parser.parse_args()

    # ─── Detect Mode ──────────────────────────────────────────────────────
    detected = detect_mode(args.embeddings_dir)
    mode = args.mode if args.mode != "auto" else detected
    if mode == "none":
        mode = "mock"  # default to mock if nothing exists

    # Set defaults for mock mode
    if args.d_model is None:
        args.d_model = 1024 if mode == "real" else 512
    if args.num_heads is None:
        args.num_heads = 16 if mode == "real" else 8

    mode_label = (
        "REAL (Qwen3-0.6B 4-bit)" if mode == "real" else "MOCK (NumPy synthetic)"
    )

    print("=" * 70)
    print("  PLAN A+B: Embedding Geometry + Epsilon Sweet Spot Experiment")
    print(f"  Bubble Transformer Research — LLM-BUBBLE ({mode_label})")
    print("=" * 70)
    if args.cost_type != "l2_sq":
        print(f"  Cost function: {args.cost_type}")
    print()

    start_time = time.time()

    # ─── Step 1: Prepare Embeddings ─────────────────────────────────────
    raw_path = Path(args.embeddings_dir) / "raw_input.npy"
    softmax_dir = Path(args.embeddings_dir) / "softmax"

    if args.skip_generation:
        print("[Step 1/4] Skipping embedding generation (using existing)...")
        print("-" * 50)
        if not raw_path.exists():
            print("[Step 1] ERROR: raw_input.npy not found!")
            sys.exit(1)
        print("[Step 1] Embeddings found, proceeding.\n")
    elif mode == "real":
        print("[Step 1/4] Real embeddings mode — checking availability...")
        print("-" * 50)
        if (
            raw_path.exists()
            and softmax_dir.exists()
            and any(softmax_dir.glob("layer_*.npy"))
        ):
            print("[Step 1] Real embeddings already exist. Using them.")
            print(f"[Step 1] Embeddings dir: {args.embeddings_dir}/\n")
        else:
            print("[Step 1] No real embeddings found. Running extract_embeddings.py...")
            print(
                "[Step 1] This will download Qwen3-0.6B (4-bit) and extract hidden states."
            )
            print()
            from extract_embeddings import main as extract_main

            extract_main()
            print()
    else:
        print(f"[Step 1/4] Generating synthetic embeddings (mock mode)...")
        print("-" * 50)
        from generate_mock_embeddings import save_mock_embeddings

        save_mock_embeddings(
            output_dir=args.embeddings_dir,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            num_heads=args.num_heads,
            seed=args.seed,
        )
        print()

    # ─── Step 2: Run Epsilon Sweep ────────────────────────────────────────
    print("[Step 2/4] Running epsilon sweep experiment...")
    print("-" * 50)
    from plateau_attention import VALID_COST_TYPES
    from epsilon_sweep import run_epsilon_sweep

    cost_types = ["l2_sq"] if args.cost_type == "l2_sq" else [args.cost_type]
    if args.cost_type == "all":
        cost_types = VALID_COST_TYPES

    sweep_results = run_epsilon_sweep(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        epsilon_values=args.epsilon_values,
        target_layers=args.target_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        cost_types=cost_types,
    )

    if not sweep_results:
        print("[Step 2] ERROR: Sweep failed!")
        sys.exit(1)
    print(f"[Step 2] Done! {len(sweep_results.get('results', []))} results collected\n")

    # ─── Step 2b: Handle Special Modes (tension, layer-selection) ────────
    # These modes have their own visualization and report generation
    if args.mode == "tension":
        print("[Step 2b/4] Running tension sweep (dual-head comparison)...")
        print("-" * 50)
        from epsilon_sweep import run_tension_sweep

        tension_results = run_tension_sweep(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            target_layers=args.target_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
        )

        if not tension_results:
            print("[Step 2b] ERROR: Tension sweep failed!")
            sys.exit(1)
        print(
            f"[Step 2b] Done! {len(tension_results.get('results', []))} results collected\n"
        )

        # Generate visualizations
        if not args.skip_visualization:
            print("[Step 3/4] Generating tension visualizations...")
            print("-" * 50)
            from visualize import plot_tension_analysis

            tension_path = Path(args.output_dir) / "tension_sweep.json"
            if tension_path.exists():
                plot_tension_analysis(str(tension_path))
                print()
            else:
                print("[Warn] tension_sweep.json not found, skipping tension plots")

        # Generate report
        print("[Step 4/4] Tension sweep complete. See results/tension_sweep.json")
        elapsed = time.time() - start_time
        print(f"\n Tension sweep finished in {elapsed:.1f}s")
        print(f" Results: {tension_path}")
        print(f" Plots: plots/")
        print()
        return  # Skip normal epsilon sweep visualizations and report

    if args.mode == "layer-selection":
        print("[Step 3/4] Running layer selection analysis...")
        print("-" * 50)
        from epsilon_sweep import run_layer_selection
        from visualize import plot_layer_selection

        # Determine best parameters from previous sweeps if available
        # Use dot_product cost (Plan E winner) and epsilon=0.001 (Plan A+B winner) by default
        best_cost = "dot_product"
        best_epsilon = 0.001

        # Try to read from existing epsilon_sweep.json to auto-determine best parameters
        sweep_json = Path(args.output_dir) / "epsilon_sweep.json"
        if sweep_json.exists():
            try:
                with open(sweep_json, "r") as f:
                    sweep_data = json.load(f)
                sweet_spot = sweep_data.get("sweet_spot", {})
                if sweet_spot.get("epsilon"):
                    best_epsilon = sweet_spot["epsilon"]
                    print(
                        f"[LayerSelect] Using epsilon from sweet spot: {best_epsilon}"
                    )
                # For cost type, check if there is a best_cost_type in sweet_spot
                if sweet_spot.get("best_cost_type"):
                    best_cost = sweet_spot["best_cost_type"]
                    print(f"[LayerSelect] Using cost type from sweet spot: {best_cost}")
            except Exception as e:
                print(f"[LayerSelect] Could not read epsilon_sweep.json: {e}")

        # Get config for layer selection settings (dual-head, alpha values)
        try:
            from config import get_config

            cfg = get_config()
        except Exception:
            cfg = None

        # Run layer selection
        ls_results = run_layer_selection(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            d_model=args.d_model,
            num_heads=args.num_heads,
            cost_type=best_cost,
            epsilon_plateau=best_epsilon,
            config=cfg,
        )

        if not ls_results:
            print("[Step 3] ERROR: Layer selection failed!")
            sys.exit(1)

        # Generate visualizations
        if not args.skip_visualization:
            print("\n[Step 4/4] Generating layer selection visualizations...")
            print("-" * 50)
            plot_layer_selection(str(Path(args.output_dir) / "layer_selection.json"))
            print()
        else:
            print("\n[Step 4/4] Skipping layer selection visualizations")

        # Generate markdown report
        print("[Step 5/5] Writing layer selection report...")
        print("-" * 50)
        report_path = Path(args.output_dir) / "layer_selection_report.md"
        ranked = ls_results.get("ranked_layers", [])
        optimal_alpha = ls_results.get("optimal_alpha")
        baseline = ls_results.get("baseline", {})
        plateau = ls_results.get("plateau", {})
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Layer Selection Report\n\n")
            f.write(f"**Experiment**: Plan D — Layer Selection\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Recommended Configuration\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| **Cost Function** | `{best_cost}` |\n")
            f.write(f"| **Epsilon** | `{best_epsilon}` |\n")
            ranked_str = ", ".join(map(str, ranked))
            f.write(f"| **Layers to Replace** | `{ranked_str}` |\n")
            if optimal_alpha is not None:
                f.write(f"| **Dual-Head α** | `{optimal_alpha}` |\n\n")
            else:
                f.write(f"| **Dual-Head α** | `N/A (use single-head)` |\n\n")
            f.write("## Evidence\n\n")
            f.write(
                "| Layer | Rank Improvement | Concentration Gain | Intrinsic Dim Pres. |\n"
            )
            f.write(
                "|-------|------------------|-------------------|--------------------|\n"
            )
            for l in ranked:
                b = baseline.get(l, {})
                p = plateau.get(l, {})
                if not b or not p:
                    continue
                rank_imp = (
                    (p["effective_rank"] - b["effective_rank"])
                    / b["effective_rank"]
                    * 100
                )
                b_conc = b.get("concentration_ratio", 0)
                p_conc = p.get("concentration_ratio", 0)
                conc_gain = (p_conc - b_conc) / max(b_conc, 1e-6) * 100
                dim_pres = p["intrinsic_dim_mle"] / b["intrinsic_dim_mle"] * 100
                f.write(
                    f"| {l} | {rank_imp:+.1f}% | {conc_gain:+.1f}% | {dim_pres:.1f}% |\n"
                )
            f.write("\n")
        print(f"[Step 5] Report saved to {report_path}\n")

        # Summary
        print("=" * 70)
        print("  LAYER SELECTION COMPLETE")
        print("=" * 70)
        print(f"\n  Recommended layers to replace: {ranked}")
        print(f"  Configuration: epsilon={best_epsilon}, cost={best_cost}")
        if optimal_alpha is not None:
            print(f"  Optional Dual-Head alpha: {optimal_alpha}")
        print(f"\n  Results: results/layer_selection.json")
        print(f"  Report:  results/layer_selection_report.md")
        print(f"  Plots:   plots/layer_selection_*.png")
        print()
        return  # Exit early; no further steps

    # ─── Step 3: Generate Visualizations ──────────────────────────────────
    if not args.skip_visualization:
        print("[Step 3/4] Generating visualizations...")
        print("-" * 50)
        from visualize import generate_all_plots

        results_path = Path(args.output_dir) / "epsilon_sweep.json"
        generate_all_plots(str(results_path))
        print()
    else:
        print("[Step 3/4] Skipping visualization generation\n")

    # ─── Step 4: Generate Sweet Spot Report ───────────────────────────────
    print("[Step 4/4] Generating sweet spot analysis report...")
    print("-" * 50)

    results_path = Path(args.output_dir) / "epsilon_sweep.json"
    with open(results_path, "r") as f:
        data = json.load(f)

    sweet_spot = data.get("sweet_spot", {})

    report_path = Path(args.output_dir) / "sweet_spot_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Sweet Spot Analysis Report\n\n")
        f.write(f"**Experiment**: Plan A+B — Embedding Geometry + Epsilon Sweet Spot\n")
        f.write(f"**Mode**: Mock (NumPy synthetic embeddings)\n")
        f.write(f"**Date**: {data.get('date', 'N/A')}\n\n")

        f.write("## Recommended Configuration\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|-----------|-------|\n")
        f.write(f"| **Optimal eps** | `{sweet_spot.get('epsilon', 'N/A')}` |\n")
        f.write(f"| **Best Layers** | `{sweet_spot.get('layers', 'N/A')}` |\n")
        f.write(f"| **Confidence** | `{sweet_spot.get('confidence', 'N/A')}` |\n\n")

        f.write("## Expected Metrics at Sweet Spot\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(
            f"| Concentration Ratio | `{sweet_spot.get('concentration_ratio', 0):.4f}` |\n"
        )
        f.write(f"| Effective Rank | `{sweet_spot.get('effective_rank', 0):.1f}` |\n")
        f.write(
            f"| Anisotropy Index | `{sweet_spot.get('anisotropy_index', 0):.4f}` |\n"
        )
        f.write(
            f"| Intrinsic Dim (MLE) | `{sweet_spot.get('intrinsic_dim_mle', 0):.1f}` |\n\n"
        )

        f.write("## All Epsilon Scores\n\n")
        f.write(
            "| eps | Concentration | Eff. Rank | Anisotropy | Intrinsic Dim | Constraints |\n"
        )
        f.write(
            "|-----|--------------|-----------|------------|---------------|-------------|\n"
        )

        all_scores = sweet_spot.get("all_epsilon_scores", {})
        for eps_str, scores in sorted(all_scores.items(), key=lambda x: float(x[0])):
            met = "OK" if scores.get("constraints_met") else "FAIL"
            f.write(
                f"| {eps_str} | {scores.get('concentration_ratio', 0):.4f} | "
                f"{scores.get('effective_rank', 0):.1f} | "
                f"{scores.get('anisotropy_index', 0):.4f} | "
                f"{scores.get('intrinsic_dim_mle', 0):.1f} | {met} |\n"
            )

        # Cost Function Comparison section (if multiple cost types were swept)
        cost_types_in_results = set(
            r.get("cost_type", "l2_sq") for r in data.get("results", [])
        )
        if len(cost_types_in_results) > 1:
            f.write("\n---\n\n## Cost Function Comparison\n\n")
            f.write(
                "| Cost Type | Best eps | Concentration | Eff. Rank | Anisotropy | Score |\n"
            )
            f.write(
                "|-----------|----------|---------------|-----------|------------|-------|\n"
            )

            # Group by cost_type and find best for each
            cost_best = {}
            for r in data.get("results", []):
                ct = r.get("cost_type", "l2_sq")
                if r.get("epsilon", 0) <= 0:
                    continue
                score = (
                    r.get("concentration_ratio", 1.0)
                    - r.get("effective_rank", 0) / 1000
                )
                if ct not in cost_best or score < cost_best[ct]["score"]:
                    cost_best[ct] = {**r, "score": score}

            for ct in sorted(cost_best.keys()):
                best = cost_best[ct]
                f.write(
                    f"| {ct} | {best.get('epsilon', 0):.3f} | "
                    f"{best.get('concentration_ratio', 0):.4f} | "
                    f"{best.get('effective_rank', 0):.1f} | "
                    f"{best.get('anisotropy_index', 0):.4f} | "
                    f"{best.get('score', 0):.4f} |\n"
                )

            # Recommendation
            if cost_best:
                best_overall = min(cost_best.values(), key=lambda x: x["score"])
                f.write(f"\n### Recommendation\n\n")
                f.write(
                    f"**{best_overall.get('cost_type', 'N/A')}** is the recommended cost function because:\n\n"
                )
                f.write(
                    f"1. It achieves the best concentration/expressivity trade-off (score: {best_overall.get('score', 0):.4f})\n"
                )
                f.write(
                    f"2. Optimal epsilon: **{best_overall.get('epsilon', 0):.3f}**\n"
                )
                f.write(
                    f"3. Concentration ratio: **{best_overall.get('concentration_ratio', 0):.4f}**\n"
                )
                f.write(
                    f"4. Effective rank: **{best_overall.get('effective_rank', 0):.1f}**\n"
                )

        f.write("\n## Interpretation\n\n")
        f.write(
            f"**eps = {sweet_spot.get('epsilon')}** is the optimal viscosity coefficient because:\n\n"
        )
        f.write(
            f"1. It achieves a concentration ratio of **{sweet_spot.get('concentration_ratio', 0):.4f}**\n"
        )
        f.write(f"   (lower = more concentrated attention)\n")
        f.write(
            f"2. Effective rank of **{sweet_spot.get('effective_rank', 0):.1f}** means embeddings\n"
        )
        f.write(f"   maintain expressivity without collapse\n")
        f.write(
            f"3. Anisotropy index of **{sweet_spot.get('anisotropy_index', 0):.4f}** indicates\n"
        )
        f.write(f"   balanced directional distribution\n")
        f.write(
            f"4. Intrinsic dimensionality of **{sweet_spot.get('intrinsic_dim_mle', 0):.1f}**\n"
        )
        f.write(f"   confirms a meaningful low-dimensional manifold\n\n")
        f.write(
            f"**Recommended layers**: {sweet_spot.get('layers')} — these layers show\n"
        )
        f.write(
            f"the best concentration/expressivity trade-off for Bubble Attention.\n"
        )

        if len(cost_types_in_results) > 1:
            f.write(
                f"\n**See also**: `plots/cost_comparison_pareto.png` for visual Pareto frontier comparison.\n"
            )

    print(f"[Step 4] Report saved to {report_path}\n")

    # ─── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    mode_tag = "REAL MODE" if mode == "real" else "MOCK MODE"
    print("=" * 70)
    print(f"  EXPERIMENT COMPLETE — {elapsed:.1f}s ({mode_tag})")
    print("=" * 70)
    print(f"\n  Optimal eps: {sweet_spot.get('epsilon')}")
    print(f"  Best layers: {sweet_spot.get('layers')}")
    print(f"  Concentration: {sweet_spot.get('concentration_ratio', 0):.4f}")
    print(f"  Effective Rank: {sweet_spot.get('effective_rank', 0):.1f}")
    print(f"\n  Results: {results_path}")
    print(f"  Report:  {report_path}")
    print(f"  Plots:   plots/")
    print()


if __name__ == "__main__":
    main()
