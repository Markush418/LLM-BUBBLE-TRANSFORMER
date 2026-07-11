"""Throughput benchmark: NumPy vs Pure PyTorch implementations.

Compares old (NumPy bridge) vs new (pure PyTorch) for each attention type.
Measures tokens/sec and peak GPU memory.
"""
import time
import torch
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def benchmark(fn, batch_size, seq_len, d_model, n_warmup=3, n_iters=10, device="cuda"):
    """Benchmark a single function."""
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(n_warmup):
        _ = fn(x)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = fn(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = (batch_size * seq_len * n_iters) / elapsed
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    return {"tokens_per_sec": tokens_per_sec, "peak_memory_mb": peak_memory_mb}


# ============================================================
# SOFTMAX baseline
# ============================================================
def make_softmax(d_model, n_heads):
    from torch.nn.functional import scaled_dot_product_attention
    head_dim = d_model // n_heads

    def fn(x):
        b, n, d = x.shape
        q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
        return scaled_dot_product_attention(q, k, v, is_causal=True)
    return fn


# ============================================================
# DELTANET — PyTorch native (from hybrid_attention_torch.py)
# ============================================================
def make_deltanet_pure(d_model, n_heads):
    from experiments.hybrid_attention_torch import DeltaNetTorch
    layer = DeltaNetTorch(d_model=d_model, num_heads=n_heads).cuda().half()

    def fn(x):
        out = layer(x)
        return out if isinstance(out, torch.Tensor) else out[0]
    return fn


# ============================================================
# DELTANET — old NumPy (from deltanet_attention.py)
# ============================================================
def make_deltanet_numpy(d_model, n_heads):
    from experiments.deltanet_attention import DeltaNetAttention
    layer = DeltaNetAttention(d_model=d_model, num_heads=n_heads)

    def fn(x):
        import numpy as np
        x_np = x.detach().cpu().numpy().astype(np.float32)
        out_np = layer.forward(x_np)
        return torch.from_numpy(out_np).cuda().half()
    return fn


# ============================================================
# SIRI — Pure PyTorch (from siri_torch.py)
# ============================================================
def make_siri_pure(d_model, n_heads, mode="classical"):
    from experiments.siri_torch import siri_classical_torch, siri_soft_blend_torch
    head_dim = d_model // n_heads

    def fn(x):
        b, n, d = x.shape
        q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
        if mode == "classical":
            out = siri_classical_torch(q, k, v, epsilon=0.1, tau_iters=5)
        elif mode == "soft":
            out = siri_soft_blend_torch(q, k, v, alpha=0.7, epsilon=0.1, tau_iters=5)
        return out.transpose(1, 2).contiguous().view(b, n, d)
    return fn


# ============================================================
# SIRI — old NumPy (from siri_soft.py)
# ============================================================
def make_siri_numpy(d_model, n_heads, mode="classical"):
    from experiments.siri_soft import siri_soft_blend, siri_chiller
    head_dim = d_model // n_heads

    def fn(x):
        b, n, d = x.shape
        q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
        scores_np = scores.detach().cpu().numpy()
        if mode == "classical":
            attn_np = siri_soft_blend(scores_np, alpha=1.0, tau_iters=5)
        elif mode == "soft":
            attn_np = siri_soft_blend(scores_np, alpha=0.7, tau_iters=5)
        attn = torch.from_numpy(attn_np).to(device=q.device, dtype=q.dtype)
        return torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
    return fn


# ============================================================
# GUMBEL-SINKHORN — Pure PyTorch (from siri_torch.py)
# ============================================================
def make_gumbel_pure(d_model, n_heads):
    from experiments.siri_torch import gumbel_sinkhorn_torch
    head_dim = d_model // n_heads

    def fn(x):
        b, n, d = x.shape
        q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
        out = gumbel_sinkhorn_torch(q, k, v, epsilon=0.1, tau=0.1, n_sinkhorn_iters=5)
        return out.transpose(1, 2).contiguous().view(b, n, d)
    return fn


# ============================================================
# GUMBEL-SINKHORN — old (from gumbel_sinkhorn.py)
# ============================================================
def make_gumbel_old(d_model, n_heads):
    from experiments.attention_variants.gumbel_sinkhorn import gumbel_sinkhorn_attention
    head_dim = d_model // n_heads

    def fn(x):
        b, n, d = x.shape
        q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
        out = gumbel_sinkhorn_attention(q, k, v, epsilon=0.1, tau=0.1, n_sinkhorn_iters=5, causal=True)
        return out.transpose(1, 2).contiguous().view(b, n, d)
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/throughput_numpy_vs_torch.json")
    args = parser.parse_args()

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"d_model={args.d_model}, heads={args.n_heads}, batch={args.batch_size}")
    print(f"Seq lenses: {args.seq_lens}")
    print()

    results = {
        "config": {
            "d_model": args.d_model, "n_heads": args.n_heads,
            "seq_lens": args.seq_lens, "batch_size": args.batch_size,
            "n_iters": args.n_iters, "gpu": torch.cuda.get_device_name(),
        },
        "benchmarks": {},
    }

    # Build all variants
    benchmarks = {
        "softmax":                 make_softmax(args.d_model, args.n_heads),
        "deltanet_numpy":          make_deltanet_numpy(args.d_model, args.n_heads),
        "deltanet_torch":          make_deltanet_pure(args.d_model, args.n_heads),
        "siri_classical_numpy":    make_siri_numpy(args.d_model, args.n_heads, "classical"),
        "siri_classical_torch":    make_siri_pure(args.d_model, args.n_heads, "classical"),
        "siri_soft_numpy":         make_siri_numpy(args.d_model, args.n_heads, "soft"),
        "siri_soft_torch":         make_siri_pure(args.d_model, args.n_heads, "soft"),
        "gumbel_numpy":            make_gumbel_old(args.d_model, args.n_heads),
        "gumbel_torch":            make_gumbel_pure(args.d_model, args.n_heads),
    }

    for name, fn in benchmarks.items():
        results["benchmarks"][name] = {}
        for sl in args.seq_lens:
            if sl > 2048 and "numpy" in name:
                results["benchmarks"][name][str(sl)] = {"error": "skipped"}
                continue
            print(f"  {name:30s} N={sl:5d} ... ", end="", flush=True)
            try:
                stats = benchmark(fn, args.batch_size, sl, args.d_model, n_iters=args.n_iters)
                results["benchmarks"][name][str(sl)] = stats
                print(f"{stats['tokens_per_sec']:>10,.0f} tok/s  {stats['peak_memory_mb']:>8,.1f} MB")
            except torch.cuda.OutOfMemoryError:
                print("OOM")
                results["benchmarks"][name][str(sl)] = {"error": "OOM"}
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"ERROR: {e}")
                results["benchmarks"][name][str(sl)] = {"error": str(e)}

    # Summary table
    print("\n" + "=" * 90)
    print("NUMPY vs PYTORCH — tokens/sec")
    print("=" * 90)
    header = f"{'Method':<30}" + "".join(f"{sl:>10}" for sl in args.seq_lens)
    print(header)
    print("-" * 90)
    for name in benchmarks:
        row = f"{name:<30}"
        for sl in args.seq_lens:
            d = results["benchmarks"].get(name, {}).get(str(sl), {})
            if "error" in d:
                row += f"{'---':>10}"
            else:
                row += f"{d['tokens_per_sec']:>10,.0f}"
        print(row)

    # Speedup summary
    print("\n" + "=" * 90)
    print("SPEEDUP: torch vs numpy (higher = torch wins)")
    print("=" * 90)
    pairs = [
        ("deltanet", "deltanet_numpy", "deltanet_torch"),
        ("siri_classical", "siri_classical_numpy", "siri_classical_torch"),
        ("siri_soft", "siri_soft_numpy", "siri_soft_torch"),
        ("gumbel", "gumbel_numpy", "gumbel_torch"),
    ]
    header = f"{'Comparison':<30}" + "".join(f"{sl:>10}" for sl in args.seq_lens)
    print(header)
    print("-" * 90)
    for label, old, new in pairs:
        row = f"{label:<30}"
        for sl in args.seq_lens:
            d_old = results["benchmarks"].get(old, {}).get(str(sl), {})
            d_new = results["benchmarks"].get(new, {}).get(str(sl), {})
            if "error" in d_old or "error" in d_new:
                row += f"{'---':>10}"
            else:
                speedup = d_new["tokens_per_sec"] / d_old["tokens_per_sec"]
                row += f"{speedup:>9.1f}x"
        print(row)
    print("=" * 90)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
