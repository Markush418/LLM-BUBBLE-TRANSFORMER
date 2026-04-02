"""
Epsilon Sweep Controller — NumPy Implementation
==================================================
Orchestrates the epsilon sweep across all target layers.
Pure NumPy — no PyTorch required.

For each (epsilon, layer) pair:
  1. Load raw embeddings at that layer
  2. Apply PlateauAttention with epsilon
  3. Compute all metrics
  4. Store results

Outputs: results/epsilon_sweep.json
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from plateau_attention import PlateauAttentionMechanism
from metrics import compute_all_metrics

EPSILON_VALUES = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
TARGET_LAYERS_MOCK = [3, 7, 11, 15, 19, 23]  # 24-layer model
TARGET_LAYERS_REAL = [3, 7, 11, 15, 19, 23, 27]  # Qwen3-0.6B: 28 layers
TARGET_LAYERS = None  # Auto-detect at runtime
DEFAULT_D_MODEL = 512
DEFAULT_NUM_HEADS = 8
TAU_ITERS = 5


def load_embeddings(embeddings_dir: str, layer_idx: int) -> Optional[np.ndarray]:
    path = Path(embeddings_dir) / "softmax" / f"layer_{layer_idx}.npy"
    if path.exists():
        return np.load(path)
    return None


def load_raw_input(embeddings_dir: str) -> Optional[np.ndarray]:
    path = Path(embeddings_dir) / "raw_input.npy"
    if path.exists():
        return np.load(path)
    return None


def run_epsilon_sweep(
    embeddings_dir: str = "embeddings",
    output_dir: str = "results",
    epsilon_values: List[float] = None,
    target_layers: List[int] = None,
    d_model: int = None,  # Auto-detect if None
    num_heads: int = None,  # Auto-detect if None
) -> Dict:
    epsilon_values = epsilon_values or EPSILON_VALUES
    if target_layers is None:
        # Auto-detect based on available layers
        softmax_dir = Path(embeddings_dir) / "softmax"
        if softmax_dir.exists():
            available = []
            for f in softmax_dir.glob("layer_*.npy"):
                try:
                    layer_num = int(f.stem.split("_")[1])
                    available.append(layer_num)
                except (ValueError, IndexError):
                    pass
            if available:
                max_layer = max(available)
                if max_layer >= 27:
                    target_layers = TARGET_LAYERS_REAL
                else:
                    target_layers = TARGET_LAYERS_MOCK
            else:
                target_layers = TARGET_LAYERS_MOCK
        else:
            target_layers = TARGET_LAYERS_MOCK

    # Auto-detect d_model and num_heads from metadata
    metadata_path = Path(embeddings_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        if d_model is None:
            d_model = meta.get("d_model", DEFAULT_D_MODEL)
        if num_heads is None:
            num_heads = meta.get("num_attention_heads", DEFAULT_NUM_HEADS)
        mode = meta.get("mode", "unknown")
        print(f"[Sweep] Detected mode: {mode} (d_model={d_model}, heads={num_heads})")
    else:
        if d_model is None:
            d_model = DEFAULT_D_MODEL
        if num_heads is None:
            num_heads = DEFAULT_NUM_HEADS

    print(f"[Sweep] Starting epsilon sweep: {epsilon_values}")
    print(f"[Sweep] Target layers: {target_layers}")

    raw_input = load_raw_input(embeddings_dir)
    if raw_input is None:
        print(
            "[Sweep] ERROR: raw_input.npy not found. Run generate_mock_embeddings.py or extract_embeddings.py first."
        )
        return {}

    # Compute baseline (softmax) metrics from existing layer embeddings
    baseline_ranks = {}
    baseline_results = []

    print("[Sweep] Computing baseline metrics...")
    for layer_idx in target_layers:
        baseline_emb = load_embeddings(embeddings_dir, layer_idx)
        if baseline_emb is not None:
            metrics = compute_all_metrics(baseline_emb)
            baseline_ranks[layer_idx] = metrics["effective_rank"]
            baseline_results.append(
                {
                    "layer": layer_idx,
                    "epsilon": 0.0,
                    **metrics,
                }
            )
            print(
                f"  Layer {layer_idx}: eff_rank={metrics['effective_rank']:.1f}, "
                f"intrinsic_dim={metrics['intrinsic_dim_mle']:.1f}"
            )

    # Run epsilon sweep
    all_results = list(baseline_results)
    total = len(epsilon_values) * len(target_layers)
    pbar = tqdm(total=total, desc="Epsilon Sweep")

    for eps in epsilon_values:
        attn = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=eps,
            tau_iters=TAU_ITERS,
        )

        for layer_idx in target_layers:
            start_time = time.time()
            try:
                output, attn_matrix = attn.forward(raw_input, return_attention=True)
                metrics = compute_all_metrics(
                    output,
                    attn_matrix,
                    baseline_effective_rank=baseline_ranks.get(layer_idx),
                )
                result = {"layer": layer_idx, "epsilon": eps, **metrics}
                all_results.append(result)

                elapsed = time.time() - start_time
                pbar.set_postfix(
                    {
                        "eps": f"{eps:.3f}",
                        "layer": layer_idx,
                        "eff_rank": f"{metrics['effective_rank']:.1f}",
                        "conc": f"{metrics.get('concentration_ratio', 0):.3f}",
                        "time": f"{elapsed:.1f}s",
                    }
                )
            except (ValueError, RuntimeError, MemoryError) as e:
                print(f"\n[Sweep] ERROR at eps={eps}, layer={layer_idx}: {e}")
                all_results.append(
                    {"layer": layer_idx, "epsilon": eps, "error": str(e)}
                )

            pbar.update(1)

    pbar.close()

    sweet_spot = identify_sweet_spot(all_results, epsilon_values, target_layers)

    full_results = {
        "experiment": "Plan A+B: Embedding Geometry + Epsilon Sweet Spot",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "mock_numpy",
        "config": {
            "epsilon_values": epsilon_values,
            "target_layers": target_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "tau_iters": TAU_ITERS,
        },
        "baseline_ranks": baseline_ranks,
        "sweet_spot": sweet_spot,
        "results": all_results,
    }

    output_path = Path(output_dir) / "epsilon_sweep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n[Sweep] Results saved to {output_path}")
    print(f"[Sweep] Total results: {len(all_results)}")
    print(f"[Sweep] Sweet spot: eps={sweet_spot.get('epsilon', 'N/A')}")
    return full_results


def identify_sweet_spot(
    results: List[Dict], epsilon_values: List[float], target_layers: List[int]
) -> Dict:
    valid_results = [r for r in results if "error" not in r and r.get("epsilon", 0) > 0]
    if not valid_results:
        return {"epsilon": None, "reason": "No valid results"}

    epsilon_scores = {}
    for eps in epsilon_values:
        eps_results = [r for r in valid_results if r["epsilon"] == eps]
        if not eps_results:
            continue

        avg_conc = np.mean([r.get("concentration_ratio", 1.0) for r in eps_results])
        avg_rank = np.mean([r.get("effective_rank", 0) for r in eps_results])
        avg_aniso = np.mean([r.get("anisotropy_index", 1.0) for r in eps_results])
        avg_dim = np.mean([r.get("intrinsic_dim_mle", 0) for r in eps_results])

        rank_ok = avg_rank >= 50
        aniso_ok = avg_aniso < 0.5
        dim_ok = avg_dim >= 20

        if rank_ok and aniso_ok and dim_ok:
            score = avg_conc
            epsilon_scores[eps] = {
                "score": score,
                "concentration_ratio": avg_conc,
                "effective_rank": avg_rank,
                "anisotropy_index": avg_aniso,
                "intrinsic_dim_mle": avg_dim,
                "constraints_met": True,
            }
        else:
            penalty = (
                (0 if rank_ok else 0.3)
                + (0 if aniso_ok else 0.2)
                + (0 if dim_ok else 0.2)
            )
            epsilon_scores[eps] = {
                "score": avg_conc + penalty,
                "concentration_ratio": avg_conc,
                "effective_rank": avg_rank,
                "anisotropy_index": avg_aniso,
                "intrinsic_dim_mle": avg_dim,
                "constraints_met": False,
            }

    if not epsilon_scores:
        return {"epsilon": None, "reason": "No epsilon passed constraints"}

    best_eps = min(epsilon_scores, key=lambda e: epsilon_scores[e]["score"])
    best = epsilon_scores[best_eps]

    eps_results = [r for r in valid_results if r["epsilon"] == best_eps]
    best_layers = sorted(
        eps_results,
        key=lambda r: (
            r.get("concentration_ratio", 1.0) - r.get("effective_rank", 0) / 1000
        ),
    )[:3]
    best_layer_indices = [r["layer"] for r in best_layers]

    return {
        "epsilon": best_eps,
        "layers": best_layer_indices,
        "concentration_ratio": float(best["concentration_ratio"]),
        "effective_rank": float(best["effective_rank"]),
        "anisotropy_index": float(best["anisotropy_index"]),
        "intrinsic_dim_mle": float(best["intrinsic_dim_mle"]),
        "confidence": "high" if best["constraints_met"] else "medium",
        "all_epsilon_scores": {
            str(k): {kk: float(vv) for kk, vv in v.items()}
            for k, v in epsilon_scores.items()
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run epsilon sweep (numpy-only)")
    parser.add_argument("--embeddings-dir", type=str, default="embeddings")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--epsilon-values", type=float, nargs="+", default=None)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    args = parser.parse_args()

    run_epsilon_sweep(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        epsilon_values=args.epsilon_values,
        target_layers=args.target_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
    )
