"""
DeltaNet Parallel Benchmark — Recurrent vs Chunkwise-Parallel vs PyTorch
=========================================================================

Benchmarks throughput (tok/s) and correctness across sequence lengths.
"""

import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from experiments.deltanet_attention import delta_rule_recurrent, delta_rule_parallel

B, H, D_H = 1, 8, 128
D_MODEL = H * D_H  # 1024
NUM_ITERATIONS = 5

def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def _make_numpy_inputs(N, device="cpu"):
    rng = np.random.RandomState(42)
    Q = rng.randn(B, H, N, D_H).astype(np.float32) * 0.1
    K = rng.randn(B, H, N, D_H).astype(np.float32) * 0.1
    V = rng.randn(B, H, N, D_H).astype(np.float32) * 0.1
    return Q, K, V

def _bench_numpy(fn, Q, K, V, iterations=NUM_ITERATIONS):
    """Benchmark a NumPy function over B*H heads, return list of per-call times."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        for b in range(B):
            for h in range(H):
                fn(Q[b, h], K[b, h], V[b, h])
        times.append(time.perf_counter() - t0)
    return times

def _bench_torch(model, seq_len, device, dtype, iterations=NUM_ITERATIONS):
    """Benchmark DeltaNetTorch forward pass."""
    import torch
    model.eval()
    x = torch.randn(B, seq_len, D_MODEL, device=device, dtype=dtype)
    with torch.no_grad():
        _sync()
        _ = model(x)
        _sync()
        times = []
        for _ in range(iterations):
            _sync()
            t0 = time.perf_counter()
            _ = model(x)
            _sync()
            times.append(time.perf_counter() - t0)
    return times, x

def run_numpy_benchmarks():
    """Benchmark NumPy recurrent vs parallel (CPU)."""
    results = {}
    for N in [256, 512, 1024, 2048, 4096]:
        Q, K, V = _make_numpy_inputs(N)

        times_rec = _bench_numpy(delta_rule_recurrent, Q, K, V)
        times_par = _bench_numpy(lambda q, k, v: delta_rule_parallel(q, k, v, chunk_size=256), Q, K, V)

        # Correctness check
        out_rec_all = []
        out_par_all = []
        for b in range(B):
            for h in range(H):
                o_rec, _ = delta_rule_recurrent(Q[b, h], K[b, h], V[b, h])
                o_par, _ = delta_rule_parallel(Q[b, h], K[b, h], V[b, h], chunk_size=256)
                out_rec_all.append(o_rec)
                out_par_all.append(o_par)
        max_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(out_rec_all, out_par_all))

        total_tokens = B * H * N
        med_rec = sorted(times_rec)[len(times_rec) // 2]
        med_par = sorted(times_par)[len(times_par) // 2]
        throughput_rec = total_tokens / med_rec
        throughput_par = total_tokens / med_par

        results[str(N)] = {
            "recurrent_median_s": round(med_rec, 6),
            "parallel_median_s": round(med_par, 6),
            "recurrent_throughput_toks": round(throughput_rec, 1),
            "parallel_throughput_toks": round(throughput_par, 1),
            "max_abs_diff": round(max_diff, 6),
        }
    return results

def run_torch_benchmarks():
    """Benchmark DeltaNetTorch (CUDA if available, else CPU)."""
    import torch
    from experiments.hybrid_attention_torch import DeltaNetTorch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = DeltaNetTorch(d_model=D_MODEL, num_heads=H).to(device=device, dtype=dtype)

    results = {}
    for N in [256, 512, 1024, 2048]:
        total_tokens = B * H * N
        times, x = _bench_torch(model, N, device, dtype)
        med = sorted(times)[len(times) // 2]
        throughput = total_tokens / med
        results[str(N)] = {
            "median_s": round(med, 6),
            "throughput_toks": round(throughput, 1),
            "device": device,
            "dtype": str(dtype),
        }
    return results

def main():
    print(json.dumps({
        "numpy_rec_vs_par": run_numpy_benchmarks(),
        "torch_deltanet": run_torch_benchmarks(),
    }, indent=2))

if __name__ == "__main__":
    main()
