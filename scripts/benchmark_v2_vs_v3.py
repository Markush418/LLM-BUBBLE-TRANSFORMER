"""
Benchmark: V2 (Sinkhorn) vs V3 (SDOT)
=====================================

Compares throughput and memory usage between:
- V2: PlateauAttentionMechanism (Sinkhorn-Knopp iterative)
- V3: SDOTAttention (Semi-Discrete Optimal Transport)

Metrics:
- Throughput: tokens/second
- Memory: peak memory allocation (MB)
- Latency: time per forward pass (ms)
"""

import sys
import os
import argparse
import time
from typing import Dict, Tuple, Optional

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.sdot_attention import SDOTAttention
from experiments.plateau_attention import PlateauAttentionMechanism


def benchmark_throughput(
    model_v2: PlateauAttentionMechanism,
    model_v3: SDOTAttention,
    input_tensor: torch.Tensor,
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: str = "cpu",
) -> Dict:
    """
    Compare throughput between V2 (Sinkhorn) and V3 (SDOT).

    Args:
        model_v2: PlateauAttentionMechanism instance
        model_v3: SDOTAttention instance
        input_tensor: Input tensor [B, N, d_model]
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Device to use

    Returns:
        {
            'v2_time_ms': float,
            'v3_time_ms': float,
            'speedup': float,
            'v2_tokens_per_sec': float,
            'v3_tokens_per_sec': float
        }
    """
    B, N, d_model = input_tensor.shape

    # Move to device
    input_tensor = input_tensor.to(device)
    model_v3 = model_v3.to(device)

    # Warmup
    for _ in range(warmup_runs):
        # V2 (NumPy-based, needs CPU)
        input_np = input_tensor.cpu().numpy().astype(np.float32)
        _ = model_v2.forward(input_np)

        # V3 (PyTorch)
        if device == "cuda":
            torch.cuda.synchronize()
        _ = model_v3(input_tensor)
        if device == "cuda":
            torch.cuda.synchronize()

    # Benchmark V2
    times_v2 = []
    for _ in range(num_runs):
        input_np = input_tensor.cpu().numpy().astype(np.float32)
        start = time.perf_counter()
        _ = model_v2.forward(input_np)
        end = time.perf_counter()
        times_v2.append((end - start) * 1000)  # ms

    # Benchmark V3
    times_v3 = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model_v3(input_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times_v3.append((end - start) * 1000)  # ms

    avg_v2 = np.mean(times_v2)
    avg_v3 = np.mean(times_v3)

    total_tokens = B * N

    return {
        "v2_time_ms": float(avg_v2),
        "v3_time_ms": float(avg_v3),
        "speedup": float(avg_v2 / avg_v3) if avg_v3 > 0 else 0.0,
        "v2_tokens_per_sec": float(total_tokens / (avg_v2 / 1000)),
        "v3_tokens_per_sec": float(total_tokens / (avg_v3 / 1000)),
    }


def benchmark_memory(
    model_v2: PlateauAttentionMechanism,
    model_v3: SDOTAttention,
    input_tensor: torch.Tensor,
    device: str = "cpu",
) -> Dict:
    """
    Compare memory usage between V2 and V3.

    Args:
        model_v2: PlateauAttentionMechanism instance
        model_v3: SDOTAttention instance
        input_tensor: Input tensor [B, N, d_model]
        device: Device to use

    Returns:
        {
            'v2_memory_mb': float,
            'v3_memory_mb': float,
            'memory_reduction': float
        }
    """
    B, N, d_model = input_tensor.shape

    # Move to device
    input_tensor = input_tensor.to(device)
    model_v3 = model_v3.to(device)

    # V2 memory (NumPy-based, estimate from attention matrix)
    # V2 creates attention matrix of size [B, heads, N, N]
    num_heads = model_v3.num_heads
    v2_attention_size = B * num_heads * N * N * 4  # float32 = 4 bytes
    v2_memory_mb = v2_attention_size / (1024 * 1024)

    # V3 memory (measure actual)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Run forward pass
        _ = model_v3(input_tensor)
        torch.cuda.synchronize()

        # Get peak memory
        v3_memory_bytes = torch.cuda.max_memory_allocated()
        v3_memory_mb = v3_memory_bytes / (1024 * 1024)
    else:
        # CPU: estimate from centroids + assignments
        # Centroids: [B, heads, C, d_head]
        # Assignments: [B, heads, N]
        C = model_v3.num_centroids
        d_head = d_model // num_heads
        centroids_size = B * num_heads * C * d_head * 4
        assignments_size = B * num_heads * N * 4
        v3_memory_mb = (centroids_size + assignments_size) / (1024 * 1024)

    return {
        "v2_memory_mb": float(v2_memory_mb),
        "v3_memory_mb": float(v3_memory_mb),
        "memory_reduction": float(v2_memory_mb / v3_memory_mb)
        if v3_memory_mb > 0
        else 0.0,
    }


def run_comprehensive_benchmark(
    d_model: int = 512,
    num_heads: int = 8,
    batch_sizes: list = None,
    seq_lengths: list = None,
    num_centroids: int = 32,
    num_runs: int = 10,
    device: str = "cpu",
    quick: bool = False,
) -> Dict:
    """
    Run comprehensive benchmark across different configurations.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        batch_sizes: List of batch sizes to test
        seq_lengths: List of sequence lengths to test
        num_centroids: Number of centroids for V3
        num_runs: Number of runs per configuration
        device: Device to use
        quick: If True, use minimal configuration

    Returns:
        Dict with benchmark results
    """
    if quick:
        batch_sizes = [2]
        seq_lengths = [128]
        num_runs = 3
    elif batch_sizes is None:
        batch_sizes = [1, 2, 4]
    if seq_lengths is None:
        seq_lengths = [128, 256, 512]

    # Create models
    model_v2 = PlateauAttentionMechanism(
        d_model=d_model,
        num_heads=num_heads,
        epsilon=0.01,
        tau_iters=5,
    )

    model_v3 = SDOTAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_centroids=num_centroids,
        use_baroreceptor=False,
    )

    results = {
        "config": {
            "d_model": d_model,
            "num_heads": num_heads,
            "num_centroids": num_centroids,
            "device": device,
        },
        "benchmarks": [],
    }

    for B in batch_sizes:
        for N in seq_lengths:
            print(f"\n[Benchmark] B={B}, N={N}")

            input_tensor = torch.randn(B, N, d_model)

            # Throughput benchmark
            throughput_results = benchmark_throughput(
                model_v2=model_v2,
                model_v3=model_v3,
                input_tensor=input_tensor,
                num_runs=num_runs,
                device=device,
            )

            # Memory benchmark
            memory_results = benchmark_memory(
                model_v2=model_v2,
                model_v3=model_v3,
                input_tensor=input_tensor,
                device=device,
            )

            benchmark_result = {
                "batch_size": B,
                "seq_length": N,
                "throughput": throughput_results,
                "memory": memory_results,
            }

            results["benchmarks"].append(benchmark_result)

            print(f"  V2 time: {throughput_results['v2_time_ms']:.2f} ms")
            print(f"  V3 time: {throughput_results['v3_time_ms']:.2f} ms")
            print(f"  Speedup: {throughput_results['speedup']:.2f}x")
            print(f"  V3 memory: {memory_results['v3_memory_mb']:.2f} MB")

    return results


def print_summary(results: Dict):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  d_model: {results['config']['d_model']}")
    print(f"  num_heads: {results['config']['num_heads']}")
    print(f"  num_centroids: {results['config']['num_centroids']}")
    print(f"  device: {results['config']['device']}")

    print("\nResults:")
    print("-" * 60)
    print(
        f"{'B':>4} {'N':>6} {'V2(ms)':>10} {'V3(ms)':>10} {'Speedup':>10} {'V3(MB)':>10}"
    )
    print("-" * 60)

    for bench in results["benchmarks"]:
        B = bench["batch_size"]
        N = bench["seq_length"]
        v2_time = bench["throughput"]["v2_time_ms"]
        v3_time = bench["throughput"]["v3_time_ms"]
        speedup = bench["throughput"]["speedup"]
        v3_mem = bench["memory"]["v3_memory_mb"]

        print(
            f"{B:>4} {N:>6} {v2_time:>10.2f} {v3_time:>10.2f} {speedup:>10.2f}x {v3_mem:>10.2f}"
        )

    print("-" * 60)

    # Compute averages
    avg_speedup = np.mean([b["throughput"]["speedup"] for b in results["benchmarks"]])
    avg_v3_mem = np.mean([b["memory"]["v3_memory_mb"] for b in results["benchmarks"]])

    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"Average V3 memory: {avg_v3_mem:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark V2 vs V3")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num-centroids", type=int, default=32, help="Number of centroids for V3"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of runs per config"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        args.device = "cpu"

    # Run benchmark
    results = run_comprehensive_benchmark(
        d_model=args.d_model,
        num_heads=args.num_heads,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_centroids=args.num_centroids,
        num_runs=args.num_runs,
        device=args.device,
        quick=args.quick,
    )

    # Print summary
    print_summary(results)

    # Save results if output path provided
    if args.output:
        import json
        from pathlib import Path

        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
