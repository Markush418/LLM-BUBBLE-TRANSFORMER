"""Throughput benchmark: Bubble vs Softmax attention.

Measures tokens/sec, peak memory, and KV cache size at multiple context lengths.
Based on HYPIC methodology (arXiv:2607.01299) and SpotAttention (arXiv:2606.22874).

Usage:
    py experiments/throughput/benchmark_throughput.py
    py experiments/throughput/benchmark_throughput.py --seq-lens 256 512 1024 2048
    py experiments/throughput/benchmark_throughput.py --d-model 512 --n-heads 8
"""
import time
import torch
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def benchmark_attention(
    attention_fn,
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_warmup: int = 3,
    n_iters: int = 10,
    device: str = "cuda",
) -> dict:
    """Benchmark a single attention function."""
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(n_warmup):
        _ = attention_fn(x)
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = attention_fn(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = (batch_size * seq_len * n_iters) / elapsed
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0
    )

    return {
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_memory_mb,
        "elapsed_sec": round(elapsed, 4),
    }


def softmax_attention(d_model: int, n_heads: int):
    """Standard softmax attention for baseline."""
    from torch.nn.functional import scaled_dot_product_attention

    head_dim = d_model // n_heads

    def fn(x):
        b, n, d = x.shape
        q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
        return scaled_dot_product_attention(q, k, v, is_causal=True)

    return fn


def delta_net_attention(d_model: int, n_heads: int):
    """DeltaNet linear attention."""
    from experiments.hybrid_attention_torch import DeltaNetTorch

    layer = DeltaNetTorch(d_model=d_model, num_heads=n_heads).cuda().half()

    def fn(x):
        out = layer(x)
        return out if isinstance(out, torch.Tensor) else out[0]

    return fn


def siri_attention(d_model: int, n_heads: int, mode: str = "classical"):
    """SIRI attention with various modes."""
    from experiments.siri_soft import siri_soft_blend, siri_chiller

    head_dim = d_model // n_heads

    def fn(x):
        b, n, d = x.shape
        q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
        # Compute scores: QK^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
        scores_np = scores.detach().cpu().numpy()
        if mode == "classical":
            # Classical SIRI = soft_blend with alpha=1.0 (pure Sinkhorn)
            attn_np = siri_soft_blend(scores_np, alpha=1.0, tau_iters=5)
        elif mode == "soft":
            attn_np = siri_soft_blend(scores_np, alpha=0.7, tau_iters=5)
        elif mode == "chiller":
            attn_np = siri_chiller(scores_np, beta=5.0, tau_iters=5)
        attn = torch.from_numpy(attn_np).to(device=q.device, dtype=q.dtype)
        return torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)

    return fn


def gumbel_sinkhorn_attention(d_model: int, n_heads: int):
    """Gumbel-Sinkhorn attention at low temperature."""
    from experiments.attention_variants.gumbel_sinkhorn import gumbel_sinkhorn_attention as gsa

    head_dim = d_model // n_heads

    def fn(x):
        b, n, d = x.shape
        q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
        out = gsa(q, k, v, epsilon=0.1, tau=0.1, n_sinkhorn_iters=5, causal=True)
        return out.transpose(1, 2).contiguous().view(b, n, d)

    return fn


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=str, default="results/throughput_benchmark.json")
    parser.add_argument("--n-iters", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Float16: supported")
    print(f"d_model: {args.d_model}, n_heads: {args.n_heads}")
    print(f"Seq lenses: {args.seq_lens}")
    print(f"Batch size: {args.batch_size}, Iters: {args.n_iters}")
    print()

    results = {
        "config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "seq_lens": args.seq_lens,
            "batch_size": args.batch_size,
            "n_iters": args.n_iters,
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else "N/A",
        },
        "benchmarks": {},
    }

    attention_types = {
        "softmax": softmax_attention(args.d_model, args.n_heads),
        "delta_net": delta_net_attention(args.d_model, args.n_heads),
        "siri_classical": siri_attention(args.d_model, args.n_heads, mode="classical"),
        "siri_soft": siri_attention(args.d_model, args.n_heads, mode="soft"),
        "gumbel_sinkhorn_tau01": gumbel_sinkhorn_attention(args.d_model, args.n_heads),
    }

    for attn_name, attn_fn in attention_types.items():
        results["benchmarks"][attn_name] = {}
        for seq_len in args.seq_lens:
            # Skip large sequences that would OOM on 4GB VRAM
            if seq_len > 2048 and attn_name != "softmax":
                print(f"  Skipping {attn_name} @ seq_len={seq_len} (likely OOM on 4GB)")
                results["benchmarks"][attn_name][str(seq_len)] = {"error": "skipped_oom"}
                continue

            print(f"Benchmarking {attn_name} @ seq_len={seq_len}...")
            try:
                stats = benchmark_attention(
                    attn_fn,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    d_model=args.d_model,
                    n_iters=args.n_iters,
                    device=device,
                )
                results["benchmarks"][attn_name][str(seq_len)] = stats
                print(f"  tokens/sec: {stats['tokens_per_sec']:>12,.1f}")
                print(f"  peak memory: {stats['peak_memory_mb']:>8,.1f} MB")
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM — skipping remaining seq_lens for {attn_name}")
                results["benchmarks"][attn_name][str(seq_len)] = {"error": "OOM"}
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"  ERROR: {e}")
                results["benchmarks"][attn_name][str(seq_len)] = {"error": str(e)}

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY (tokens/sec)")
    print("=" * 70)
    header = f"{'Attention':<25}" + "".join(f"{sl:>10}" for sl in args.seq_lens)
    print(header)
    print("-" * 70)
    for attn_name in attention_types:
        row = f"{attn_name:<25}"
        for seq_len in args.seq_lens:
            data = results["benchmarks"].get(attn_name, {}).get(str(seq_len), {})
            if "error" in data:
                row += f"{'---':>10}"
            else:
                row += f"{data['tokens_per_sec']:>10,.0f}"
        print(row)
    print("=" * 70)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
