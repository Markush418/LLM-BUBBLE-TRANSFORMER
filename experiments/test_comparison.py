"""
Test Comparison Benchmark — Compare Attention Mechanisms
=============================================================
Benchmarks different attention mechanisms side-by-side:
- Softmax (baseline)
- PlateauAttention (Sinkhorn)
- GOATAttention (gated)

Metrics compared:
- Concentration ratio
- Effective rank
- Spectral log-det
- Attention entropy
- Forward pass time

Usage:
    python test_comparison.py
    python test_comparison.py --epochs 1000
    python test_comparison.py --output results/benchmark.json
"""

import argparse
import json
import time
import numpy as np
from typing import Dict, List, Callable
from .metrics import compute_all_metrics
from .plateau_attention import PlateauAttentionMechanism, L2SquaredCost
from .goat_attention import GOATAttentionMechanism


def generate_test_data(
    batch: int = 4, seq_len: int = 64, d_model: int = 256, seed: int = 42
) -> np.ndarray:
    """Generate synthetic test embeddings."""
    np.random.seed(seed)
    x = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    # Normalize to unit sphere
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    return x


def benchmark_mechanism(
    name: str,
    factory: Callable,
    embeddings: np.ndarray,
    num_runs: int = 10,
) -> Dict:
    """Benchmark a single attention mechanism."""
    # Warm-up
    _ = factory().forward(embeddings)

    times = []
    outputs = []

    for _ in range(num_runs):
        mech = factory()

        start = time.perf_counter()
        out = mech.forward(embeddings)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        outputs.append(out)

    # Compute metrics on last output
    metrics = compute_all_metrics(outputs[-1])
    metrics["name"] = name
    metrics["time_mean"] = np.mean(times)
    metrics["time_std"] = np.std(times)

    return metrics


def run_comparison(
    d_model: int = 256,
    num_heads: int = 4,
    seq_len: int = 64,
    batch: int = 4,
    epsilon: float = 0.1,
    num_runs: int = 10,
    verbose: bool = True,
) -> List[Dict]:
    """Run full comparison across mechanisms."""

    embeddings = generate_test_data(batch, seq_len, d_model)

    results = []

    # 1. Softmax baseline (simplified - just linear if no QK scale)
    if verbose:
        print("=" * 50)
        print("  Running benchmarks...")
        print("=" * 50)

    # Softmax approximation via PyTorch-like scale
    def softmax_factory():
        return PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=10.0,  # Very high epsilon = close to softmax
            tau_iters=5,
        )

    results.append(
        benchmark_mechanism("Softmax (high eps)", softmax_factory, embeddings, num_runs)
    )
    if verbose:
        print(f"  [1/4] Softmax...")

    # 2. PlateauAttention (standard)
    def plateau_factory():
        return PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon,
            tau_iters=5,
        )

    results.append(
        benchmark_mechanism(
            f"Plateau (ε={epsilon})", plateau_factory, embeddings, num_runs
        )
    )
    if verbose:
        print(f"  [2/4] Plateau (ε={epsilon})...")

    # 3. GOATAttention
    def goat_factory():
        return GOATAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon,
            learn_gates=True,
        )

    results.append(
        benchmark_mechanism(f"GOAT (ε={epsilon})", goat_factory, embeddings, num_runs)
    )
    if verbose:
        print(f"  [3/4] GOAT...")

    # 4. Dual-head (low epsilon)
    def plateau_low_factory():
        return PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=0.01,
            tau_iters=5,
        )

    results.append(
        benchmark_mechanism(
            "Plateau (ε=0.01)", plateau_low_factory, embeddings, num_runs
        )
    )
    if verbose:
        print(f"  [4/4] Low epsilon...")

    return results


def print_comparison(results: List[Dict]) -> None:
    """Print comparison in table format."""
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    # Header
    print(
        f"{'Mechanism':<25} {'Time (ms)':<12} {'Eff Rank':<10} {'Ent':<8} {'Conc':<8}"
    )
    print("-" * 70)

    for r in results:
        name = r.get("name", "unknown")[:24]
        time_ms = r.get("time_mean", 0) * 1000
        eff_rank = r.get("effective_rank", 0)
        ent = r.get("attention_entropy", r.get("spectral_log_det", 0))
        conc = r.get("concentration_ratio", 0)

        print(
            f"{name:<25} {time_ms:>8.2f}    {eff_rank:>7.1f}  {ent:>6.2f}  {conc:>6.3f}"
        )

    print("-" * 70)

    # Speedup vs baseline
    fast = results[0].get("time_mean", 1)
    for r in results[1:]:
        speedup = fast / r.get("time_mean", 1)
        name = r.get("name", "unknown")[:24]
        print(f"  {name}: {speedup:.2f}x vs baseline")


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention mechanisms")
    parser.add_argument("--d-model", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value")
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of benchmark runs"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--verbose", type=int, default=1, help="Verbose output")

    args = parser.parse_args()

    results = run_comparison(
        d_model=args.d_model,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        batch=args.batch,
        epsilon=args.epsilon,
        num_runs=args.num_runs,
        verbose=bool(args.verbose),
    )

    print_comparison(results)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
