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

from plateau_attention import PlateauAttentionMechanism, DualHeadPlateauAttention
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
        raw = np.load(path)
        # Handle 2D embeddings from real extraction: (total_tokens, D) -> (1, N, D)
        if raw.ndim == 2:
            raw = raw.reshape(1, raw.shape[0], raw.shape[1])
        return raw
    return None


def run_epsilon_sweep(
    embeddings_dir: str = "embeddings",
    output_dir: str = "results",
    epsilon_values: List[float] = None,
    target_layers: List[int] = None,
    d_model: int = None,  # Auto-detect if None
    num_heads: int = None,  # Auto-detect if None
    cost_types: List[str] = None,  # Cost function types (currently uses l2_sq)
) -> Dict:
    epsilon_values = epsilon_values or EPSILON_VALUES
    cost_types = cost_types or ["l2_sq"]  # Currently only l2_sq supported
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
    mode = "mock_numpy"  # Default mode
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        if d_model is None:
            d_model = meta.get("d_model", DEFAULT_D_MODEL)
        if num_heads is None:
            num_heads = meta.get("num_attention_heads", DEFAULT_NUM_HEADS)
        mode = meta.get("mode", "mock_numpy")
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
        "mode": mode,
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


def run_tension_sweep(
    embeddings_dir: str = "embeddings",
    output_dir: str = "results",
    target_layers: List[int] = None,
    d_model: int = None,
    num_heads: int = None,
    epsilon_low: float = 0.001,
    epsilon_high: float = 0.1,
    alpha_values: List[float] = None,
) -> Dict:
    """
    Run dual-head tension sweep comparing single vs dual-head PlateauAttention.

    For each (layer, alpha) pair:
    1. Load raw_input embeddings
    2. Run single-head PlateauAttention (epsilon_low as baseline)
    3. Run dual-head DualHeadPlateauAttention (epsilon_low, epsilon_high, alpha)
    4. Compare metrics

    Returns:
        Dict with results saved to tension_sweep.json
    """
    alpha_values = alpha_values or [0.0, 0.25, 0.5, 0.75, 1.0]

    # Auto-detect target_layers
    if target_layers is None:
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
                target_layers = (
                    TARGET_LAYERS_REAL if max_layer >= 27 else TARGET_LAYERS_MOCK
                )
            else:
                target_layers = TARGET_LAYERS_MOCK
        else:
            target_layers = TARGET_LAYERS_MOCK

    # Auto-detect d_model and num_heads from metadata
    metadata_path = Path(embeddings_dir) / "metadata.json"
    mode = "mock_numpy"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        if d_model is None:
            d_model = meta.get("d_model", DEFAULT_D_MODEL)
        if num_heads is None:
            num_heads = meta.get("num_attention_heads", DEFAULT_NUM_HEADS)
        mode = meta.get("mode", "mock_numpy")
        print(f"[Tension] Detected mode: {mode} (d_model={d_model}, heads={num_heads})")
    else:
        if d_model is None:
            d_model = DEFAULT_D_MODEL
        if num_heads is None:
            num_heads = DEFAULT_NUM_HEADS

    print(f"[Tension] Starting tension sweep")
    print(f"[Tension] Target layers: {target_layers}")
    print(f"[Tension] Epsilon low: {epsilon_low}, Epsilon high: {epsilon_high}")
    print(f"[Tension] Alpha values: {alpha_values}")

    raw_input = load_raw_input(embeddings_dir)
    if raw_input is None:
        print("[Tension] ERROR: raw_input.npy not found.")
        return {}

    all_results = []
    total = len(target_layers) * (1 + len(alpha_values))  # single + dual for each alpha
    pbar = tqdm(total=total, desc="Tension Sweep")

    # Compute baseline (softmax) metrics
    baseline_ranks = {}
    print("[Tension] Computing baseline metrics...")
    for layer_idx in target_layers:
        baseline_emb = load_embeddings(embeddings_dir, layer_idx)
        if baseline_emb is not None:
            metrics = compute_all_metrics(baseline_emb)
            baseline_ranks[layer_idx] = metrics["effective_rank"]
            print(f"  Layer {layer_idx}: eff_rank={metrics['effective_rank']:.1f}")

    # Run single-head as baseline for comparison
    single_head_results = {}
    print("\n[Tension] Running single-head baseline...")
    single_attn = PlateauAttentionMechanism(
        d_model=d_model,
        num_heads=num_heads,
        epsilon=epsilon_low,
        tau_iters=TAU_ITERS,
    )

    for layer_idx in target_layers:
        start_time = time.time()
        try:
            output, attn_matrix = single_attn.forward(raw_input, return_attention=True)
            metrics = compute_all_metrics(
                output,
                attn_matrix,
                baseline_effective_rank=baseline_ranks.get(layer_idx),
            )

            result = {
                "layer": layer_idx,
                "mode": "single",
                "epsilon": epsilon_low,
                **metrics,
            }
            all_results.append(result)
            single_head_results[layer_idx] = metrics

            elapsed = time.time() - start_time
            pbar.set_postfix(
                {
                    "mode": "single",
                    "layer": layer_idx,
                    "eff_rank": f"{metrics['effective_rank']:.1f}",
                    "time": f"{elapsed:.1f}s",
                }
            )
        except Exception as e:
            print(f"\n[Tension] ERROR single-head layer={layer_idx}: {e}")
            all_results.append(
                {
                    "layer": layer_idx,
                    "mode": "single",
                    "error": str(e),
                }
            )

        pbar.update(1)

    # Run dual-head for each alpha
    print("\n[Tension] Running dual-head experiments...")
    for alpha in alpha_values:
        dual_attn = DualHeadPlateauAttention(
            d_model=d_model,
            num_heads=num_heads,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            alpha=alpha,
            tau_iters=TAU_ITERS,
        )

        for layer_idx in target_layers:
            start_time = time.time()
            try:
                output, A_low, A_high = dual_attn.forward(
                    raw_input, return_attention=True
                )

                # Compute metrics for output
                metrics = compute_all_metrics(
                    output,
                    A_low,  # Use A_low as primary attention matrix for metrics
                    baseline_effective_rank=baseline_ranks.get(layer_idx),
                )

                # Compute entropy for both heads
                def attention_entropy(A):
                    """Compute entropy of attention distribution."""
                    # A: (batch, heads, seq, seq)
                    eps = 1e-10
                    A_safe = np.clip(A, eps, 1.0)
                    ent = -np.sum(
                        A_safe * np.log(A_safe + eps), axis=-1
                    )  # (batch, heads, seq)
                    return float(np.mean(ent))

                entropy_low = attention_entropy(A_low)
                entropy_high = attention_entropy(A_high)

                result = {
                    "layer": layer_idx,
                    "mode": "dual",
                    "alpha": alpha,
                    "epsilon_low": epsilon_low,
                    "epsilon_high": epsilon_high,
                    **metrics,
                    "entropy_A_low": entropy_low,
                    "entropy_A_high": entropy_high,
                    "entropy_ratio": entropy_high / (entropy_low + 1e-10),
                }
                all_results.append(result)

                elapsed = time.time() - start_time
                pbar.set_postfix(
                    {
                        "mode": "dual",
                        "alpha": f"{alpha:.2f}",
                        "layer": layer_idx,
                        "eff_rank": f"{metrics['effective_rank']:.1f}",
                        "time": f"{elapsed:.1f}s",
                    }
                )
            except Exception as e:
                print(
                    f"\n[Tension] ERROR dual-head alpha={alpha}, layer={layer_idx}: {e}"
                )
                all_results.append(
                    {
                        "layer": layer_idx,
                        "mode": "dual",
                        "alpha": alpha,
                        "error": str(e),
                    }
                )

            pbar.update(1)

    pbar.close()

    # Identify best alpha per layer
    comparison = analyze_tension_results(all_results, target_layers, alpha_values)

    full_results = {
        "experiment": "Dual-Head Tension Sweep",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "config": {
            "target_layers": target_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "epsilon_low": epsilon_low,
            "epsilon_high": epsilon_high,
            "alpha_values": alpha_values,
            "tau_iters": TAU_ITERS,
        },
        "baseline_ranks": baseline_ranks,
        "comparison": comparison,
        "results": all_results,
    }

    output_path = Path(output_dir) / "tension_sweep.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n[Tension] Results saved to {output_path}")
    print(f"[Tension] Total results: {len(all_results)}")
    print(f"[Tension] Comparison summary:")
    for layer, best_alpha in comparison.get("best_alpha_per_layer", {}).items():
        print(f"  Layer {layer}: best_alpha={best_alpha}")

    return full_results


def analyze_tension_results(
    results: List[Dict], target_layers: List[int], alpha_values: List[float]
) -> Dict:
    """Analyze tension sweep results to find best alpha per layer."""
    valid_results = [r for r in results if "error" not in r]

    best_alpha_per_layer = {}
    layers_where_dual_wins = []

    for layer in target_layers:
        layer_results = [r for r in valid_results if r["layer"] == layer]

        # Find single-head result
        single_result = next((r for r in layer_results if r["mode"] == "single"), None)

        # Find best dual-head result
        dual_results = [r for r in layer_results if r["mode"] == "dual"]

        if single_result and dual_results:
            # Compare by concentration ratio (lower is better for concentration)
            best_dual = min(
                dual_results, key=lambda r: r.get("concentration_ratio", 1.0)
            )

            # Check if dual is better than single
            if best_dual.get("concentration_ratio", 1.0) < single_result.get(
                "concentration_ratio", 1.0
            ):
                layers_where_dual_wins.append(layer)
                best_alpha_per_layer[layer] = best_dual["alpha"]
            else:
                best_alpha_per_layer[layer] = None  # single-head wins

    return {
        "layers_where_dual_wins": layers_where_dual_wins,
        "best_alpha_per_layer": best_alpha_per_layer,
        "total_layers": len(target_layers),
        "dual_win_count": len(layers_where_dual_wins),
    }


def find_optimal_alpha(
    tension_results: List[Dict], min_effective_rank_ratio: float = 0.5
) -> Dict:
    """Find the optimal tension coefficient alpha that maximizes concentration
    while maintaining expressivity.

    Score formula:
        score = avg_concentration * avg_tension_balance
                + 0.1 * min(avg_effective_rank / 500.0, 1.0)

    The first term rewards high concentration together with balanced tension.
    The second term provides a small bonus for preserving expressivity
    (effective_rank), capped at 500.

    Args:
        tension_results: List of result dicts from run_tension_sweep().
            Each dict must contain at minimum: "alpha", "concentration_ratio",
            "effective_rank". "tension_balance" is optional.
        min_effective_rank_ratio: Minimum ratio of effective rank to consider
            (default 0.5). Not currently enforced in scoring but reserved
            for future filtering.

    Returns:
        Dict with keys: alpha, score, avg_concentration, avg_effective_rank,
        avg_tension_balance, num_layers. If tension_results is empty, returns
        {"alpha": 0.5, "score": 0.0, "reason": "no results"}.
    """
    if not tension_results:
        return {"alpha": 0.5, "score": 0.0, "reason": "no results"}

    # Group results by alpha
    groups: Dict[float, List[Dict]] = {}
    for r in tension_results:
        alpha = r.get("alpha")
        if alpha is None:
            continue
        groups.setdefault(alpha, []).append(r)

    if not groups:
        return {"alpha": 0.5, "score": 0.0, "reason": "no results"}

    best_alpha = 0.5
    best_score = -float("inf")
    best_stats = {}

    for alpha, results in groups.items():
        concentrations = [
            r.get("concentration_ratio", 0.0) for r in results if "concentration_ratio" in r
        ]
        ranks = [
            r.get("effective_rank", 0.0) for r in results if "effective_rank" in r
        ]
        tensions = [
            r.get("tension_balance", 0.0) for r in results if "tension_balance" in r
        ]

        if not concentrations or not ranks:
            continue

        avg_concentration = float(np.mean(concentrations))
        avg_rank = float(np.mean(ranks))
        avg_tension = float(np.mean(tensions)) if tensions else 0.0

        score = (
            avg_concentration * avg_tension
            + 0.1 * min(avg_rank / 500.0, 1.0)
        )

        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_stats = {
                "avg_concentration": avg_concentration,
                "avg_effective_rank": avg_rank,
                "avg_tension_balance": avg_tension,
                "num_layers": len(results),
            }

    return {
        "alpha": best_alpha,
        "score": best_score,
        **best_stats,
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
