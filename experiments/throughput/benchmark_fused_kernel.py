"""Benchmark: Fused SIRI kernel vs NumPy vs Softmax.

Tests the new kernels/siri_fused.py against the old siri_soft.py (NumPy)
and PyTorch softmax baseline.
"""
import time
import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def benchmark(fn, seq_len, d_model, n_warmup=5, n_iters=20):
    x = torch.randn(1, seq_len, d_model, device="cuda", dtype=torch.float16)
    for _ in range(n_warmup):
        _ = fn(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = fn(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "tokens_per_sec": (seq_len * n_iters) / elapsed,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    d_model, n_heads = 1024, 16
    head_dim = d_model // n_heads
    seq_lens = [256, 512, 1024, 2048]
    
    # --- Softmax baseline ---
    from torch.nn.functional import scaled_dot_product_attention
    def make_softmax():
        def fn(x):
            b, n, _ = x.shape
            q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
            return scaled_dot_product_attention(q, k, v, is_causal=True)
        return fn
    
    # --- Fused SIRI (new PyTorch) ---
    from kernels.siri_fused import fused_siri_attention
    def make_siri_fused():
        def fn(x):
            b, n, _ = x.shape
            q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
            out = fused_siri_attention(q, k, v, epsilon=0.1, n_iters=5, mode="torch")
            return out.transpose(1, 2).contiguous().view(b, n, d_model)
        return fn
    
    # --- Fused SIRI with Power Diagram ---
    def make_siri_psi():
        psi_param = torch.randn(1, n_heads, 1, 1, device="cuda", dtype=torch.float16) * 0.1
        def fn(x):
            b, n, _ = x.shape
            q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
            psi = psi_param.expand(b, n_heads, n, 1).squeeze(-1)
            out = fused_siri_attention(q, k, v, epsilon=0.1, psi=psi, n_iters=5, mode="torch")
            return out.transpose(1, 2).contiguous().view(b, n, d_model)
        return fn
    
    # --- Chunked SIRI (memory-efficient) ---
    def make_siri_chunked():
        def fn(x):
            b, n, _ = x.shape
            q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
            out = fused_siri_attention(q, k, v, epsilon=0.1, n_iters=5, mode="chunked")
            return out.transpose(1, 2).contiguous().view(b, n, d_model)
        return fn
    
    # --- Old NumPy SIRI ---
    def make_siri_numpy():
        from experiments.siri_soft import siri_soft_blend
        def fn(x):
            b, n, _ = x.shape
            q = k = v = x.view(b, n, n_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
            scores_np = scores.detach().cpu().numpy()
            attn_np = siri_soft_blend(scores_np, alpha=1.0, tau_iters=5)
            attn = torch.from_numpy(attn_np).to(device="cuda", dtype=torch.float16)
            return torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d_model)
        return fn
    
    # --- DeltaNet (PyTorch) ---
    def make_deltanet():
        from experiments.hybrid_attention_torch import DeltaNetTorch
        layer = DeltaNetTorch(d_model=d_model, num_heads=n_heads).cuda().half()
        def fn(x):
            out = layer(x)
            return out if isinstance(out, torch.Tensor) else out[0]
        return fn
    
    variants = {
        "Softmax": make_softmax(),
        "SIRI fused (PyTorch)": make_siri_fused(),
        "SIRI fused + psi": make_siri_psi(),
        "SIRI chunked (O(N·chunk))": make_siri_chunked(),
        "SIRI old (NumPy)": make_siri_numpy(),
        "DeltaNet (PyTorch)": make_deltanet(),
    }
    
    results = {}
    for name, fn in variants.items():
        results[name] = {}
        for sl in seq_lens:
            print(f"  {name:30s} N={sl:5d} ... ", end="", flush=True)
            try:
                stats = benchmark(fn, sl, d_model)
                results[name][str(sl)] = stats
                print(f"{stats['tokens_per_sec']:>10,.0f} tok/s  {stats['peak_memory_mb']:>8,.1f} MB")
            except torch.cuda.OutOfMemoryError:
                print("OOM")
                results[name][str(sl)] = {"error": "OOM"}
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"ERROR: {e}")
                results[name][str(sl)] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("FUSED SIRI vs NUMPY vs SOFTMAX — tokens/sec")
    print("=" * 80)
    header = f"{'Method':<30}" + "".join(f"{sl:>10}" for sl in seq_lens)
    print(header)
    print("-" * 80)
    for name in variants:
        row = f"{name:<30}"
        for sl in seq_lens:
            d = results[name].get(str(sl), {})
            if "error" in d:
                row += f"{'---':>10}"
            else:
                row += f"{d['tokens_per_sec']:>10,.0f}"
        print(row)
    
    # Speedup vs NumPy
    print("\n" + "=" * 80)
    print("SPEEDUP vs NumPy baseline")
    print("=" * 80)
    for name in variants:
        if "NumPy" in name:
            continue
        row = f"{name:<30}"
        for sl in seq_lens:
            d_new = results[name].get(str(sl), {})
            d_old = results["SIRI old (NumPy)"].get(str(sl), {})
            if "error" in d_new or "error" in d_old:
                row += f"{'---':>10}"
            else:
                speedup = d_new["tokens_per_sec"] / d_old["tokens_per_sec"]
                row += f"{speedup:>9.1f}x"
        print(row)
    print("=" * 80)
    
    # Save
    out_path = Path("results/fused_kernel_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": {"d_model": d_model, "n_heads": n_heads, "seq_lens": seq_lens}}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
