"""
HybridAttention Experiment Orchestrator (Plan A+B — post-SDOT)
================================================================

Runs the full embedding geometry analysis using the new HybridAttention
(DeltaNet + SIRI + Power Diagram psi) instead of the legacy PlateauAttention.

Compatible API with run_experiment.py, but with these new flags:
  --attention-type {plateau, hybrid}   default: hybrid
  --lam {0.0..1.0}                    interpolation DeltaNet<->SIRI
  --use-psi / --no-psi                enable/disable Power Diagram bias
  --epsilon-values                    standard sweep
  --target-layers                     layers to analyze
  --d-model, --num-heads              model dimensions

Usage:
    python experiments/run_hybrid_experiment.py --mode mock
    python experiments/run_hybrid_experiment.py --mode mock --lam 0.5 --epsilon-values 0.001 0.01 0.1
    python experiments/run_hybrid_experiment.py --mode mock --attention-type plateau  # for comparison
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def detect_mode(embeddings_dir: str) -> str:
    """Auto-detect whether we have real or mock embeddings."""
    metadata_path = Path(embeddings_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        return meta.get("mode", "mock")
    softmax_dir = Path(embeddings_dir) / "softmax"
    if softmax_dir.exists() and any(softmax_dir.glob("layer_*.npy")):
        return "mock"
    return "none"


def load_embeddings(embeddings_dir: str, layer_idx: int) -> Optional[np.ndarray]:
    path = Path(embeddings_dir) / "softmax" / f"layer_{layer_idx}.npy"
    if path.exists():
        return np.load(path)
    return None


def load_raw_input(embeddings_dir: str) -> Optional[np.ndarray]:
    path = Path(embeddings_dir) / "raw_input.npy"
    if path.exists():
        raw = np.load(path)
        if raw.ndim == 2:
            raw = raw.reshape(1, raw.shape[0], raw.shape[1])
        return raw
    return None


def build_attention(args, d_model: int, num_heads: int):
    """Build the attention module per --attention-type."""
    if args.attention_type == "hybrid":
        from hybrid_attention import HybridAttention
        return HybridAttention(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=args.epsilon if hasattr(args, "epsilon") else 0.1,
            tau_iters=5,
            lam=args.lam,
            seed=args.seed,
        )
    else:
        from plateau_attention import PlateauAttentionMechanism
        return PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=args.epsilon if hasattr(args, "epsilon") else 0.1,
            tau_iters=5,
            seed=args.seed,
        )


def run_sweep(
    raw_input: np.ndarray,
    target_layers: List[int],
    epsilon_values: List[float],
    args,
) -> List[Dict]:
    """Run the attention sweep across (epsilon, layer)."""
    from metrics import compute_all_metrics

    all_results = []

    # Baseline (softmax) metrics from layer embeddings
    baseline_ranks = {}
    for layer_idx in target_layers:
        baseline_emb = load_embeddings(args.embeddings_dir, layer_idx)
        if baseline_emb is not None:
            m = compute_all_metrics(baseline_emb)
            baseline_ranks[layer_idx] = m["effective_rank"]

    # If raw_input shape is (1, N, D), we use the SAME attention for all layers
    # (since the new architecture operates on raw_input, not per-layer embeddings).
    print(f"[Sweep] raw_input shape: {raw_input.shape}")
    print(f"[Sweep] Attention type: {args.attention_type}")
    print(f"[Sweep] Lambda (hybrid only): {args.lam}")
    print(f"[Sweep] Epsilon values: {epsilon_values}")
    print(f"[Sweep] Target layers: {target_layers}")

    for eps in epsilon_values:
        args.epsilon = eps
        attn = build_attention(args, d_model=raw_input.shape[-1], num_heads=8)

        for layer_idx in target_layers:
            t0 = time.time()
            try:
                output, attn_matrix = attn(raw_input, return_attention=True)
                metrics = compute_all_metrics(
                    output,
                    attn_matrix,
                    baseline_effective_rank=baseline_ranks.get(layer_idx),
                )
                result = {
                    "layer": layer_idx,
                    "epsilon": eps,
                    "attention_type": args.attention_type,
                    "lam": args.lam if args.attention_type == "hybrid" else None,
                    "elapsed_s": time.time() - t0,
                    **metrics,
                }
                all_results.append(result)
                print(
                    f"  eps={eps:.3f}, layer={layer_idx}: "
                    f"eff_rank={metrics['effective_rank']:.1f}, "
                    f"conc={metrics.get('concentration_ratio', 0):.3f}"
                )
            except Exception as e:
                print(f"  ERROR eps={eps}, layer={layer_idx}: {e}")
                all_results.append({
                    "layer": layer_idx, "epsilon": eps, "error": str(e),
                })

    return all_results, baseline_ranks


def identify_sweet_spot(results: List[Dict], epsilon_values: List[float]) -> Dict:
    """Find the epsilon that maximizes concentration without collapse."""
    valid = [r for r in results if "error" not in r and "effective_rank" in r]
    if not valid:
        return {}

    best = None
    best_score = -1.0
    for r in valid:
        rank = r["effective_rank"]
        conc = r.get("concentration_ratio", 0)
        # Score: concentration (high) + rank above baseline (preserved)
        baseline = r.get("effective_rank_ratio", 1.0)
        if baseline > 0.5:
            score = conc * (baseline ** 0.5)
        else:
            score = 0.0
        if score > best_score:
            best_score = score
            best = r

    if best:
        return {
            "epsilon": best.get("epsilon"),
            "layer": best.get("layer"),
            "effective_rank": best.get("effective_rank"),
            "concentration_ratio": best.get("concentration_ratio"),
            "intrinsic_dim_mle": best.get("intrinsic_dim_mle"),
            "score": best_score,
            "attention_type": best.get("attention_type"),
            "lam": best.get("lam"),
        }
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="HybridAttention Experiment: DeltaNet + SIRI + Power Diagram"
    )
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "mock", "real"])
    parser.add_argument("--attention-type", type=str, default="hybrid",
                        choices=["plateau", "hybrid"],
                        help="plateau = legacy SIRI, hybrid = DeltaNet+SIRI+psi")
    parser.add_argument("--lam", type=float, default=0.5,
                        help="Hybrid interpolation DeltaNet<->SIRI (0=SIRI, 1=DeltaNet)")
    parser.add_argument("--use-psi", action="store_true", default=True,
                        help="Use Power Diagram psi bias")
    parser.add_argument("--no-psi", action="store_false", dest="use_psi")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--embeddings-dir", type=str, default="embeddings")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--epsilon-values", type=float, nargs="+", default=None)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Single epsilon (used by build_attention)")

    args = parser.parse_args()

    # Resolve mode
    if args.mode == "auto":
        args.mode = detect_mode(args.embeddings_dir)
    if args.mode == "none":
        print("[ERROR] No embeddings found. Run with --skip-generation false.")
        sys.exit(1)
    print(f"[Run] Mode: {args.mode}, Attention: {args.attention_type}, lam: {args.lam}")

    # Default epsilon sweep
    epsilon_values = args.epsilon_values or [0.001, 0.01, 0.1, 1.0]
    target_layers = args.target_layers or [3, 7, 11, 15, 19, 23]

    # Generate embeddings if needed
    if args.mode == "mock" and not args.skip_generation:
        emb_dir = Path(args.embeddings_dir)
        if not (emb_dir / "raw_input.npy").exists():
            print("[Step 1/3] Generating mock embeddings...")
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

    # Step 2: Run sweep
    print(f"\n[Step 2/3] Running {args.attention_type} sweep...")
    raw_input = load_raw_input(args.embeddings_dir)
    if raw_input is None:
        print("[ERROR] raw_input.npy not found.")
        sys.exit(1)

    results, baseline_ranks = run_sweep(
        raw_input=raw_input,
        target_layers=target_layers,
        epsilon_values=epsilon_values,
        args=args,
    )
    if not results:
        print("[ERROR] Sweep produced no results.")
        sys.exit(1)

    sweet_spot = identify_sweet_spot(results, epsilon_values)

    full_results = {
        "experiment": f"HybridAttention ({args.attention_type}) — DeltaNet + SIRI + psi",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "attention_type": args.attention_type,
        "lam": args.lam if args.attention_type == "hybrid" else None,
        "config": {
            "epsilon_values": epsilon_values,
            "target_layers": target_layers,
            "tau_iters": 5,
            "seed": args.seed,
        },
        "baseline_ranks": baseline_ranks,
        "sweet_spot": sweet_spot,
        "results": results,
    }

    # Save JSON
    output_path = Path(args.output_dir) / f"hybrid_{args.attention_type}_sweep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n[Sweep] Results saved to {output_path}")

    # Save Markdown summary
    md_path = Path(args.output_dir) / f"hybrid_{args.attention_type}_sweep.md"
    with open(md_path, "w") as f:
        f.write(f"# HybridAttention Sweep ({args.attention_type})\n\n")
        f.write(f"**Mode**: {args.mode}  \n")
        f.write(f"**Lambda**: {args.lam}  \n")
        f.write(f"**Epsilon values**: {epsilon_values}  \n")
        f.write(f"**Target layers**: {target_layers}  \n\n")
        f.write("## Sweet Spot\n\n")
        for k, v in sweet_spot.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Results by epsilon\n\n")
        f.write("| eps | layer | eff_rank | intrinsic_dim | anisotropy | concentration | entropy |\n")
        f.write("|---|-------|----------|---------------|-------------|---------------|---------|\n")
        for r in results:
            if "error" not in r:
                f.write(
                    f"| {r['epsilon']:.3f} | {r['layer']} | "
                    f"{r['effective_rank']:.1f} | "
                    f"{r['intrinsic_dim_mle']:.1f} | "
                    f"{r['anisotropy_index']:.3f} | "
                    f"{r.get('concentration_ratio', 0):.3f} | "
                    f"{r.get('attention_entropy', 0):.2f} |\n"
                )
    print(f"[Sweep] Markdown report saved to {md_path}")

    print(f"\n[Sweep] Total results: {len(results)}")
    print(f"[Sweep] Sweet spot: {sweet_spot}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()