"""Lambda sweep on REAL Qwen3-0.6B embeddings - simplified."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, "experiments")
import numpy as np
from hybrid_attention import HybridAttention
from metrics import effective_rank, intrinsic_dimension_mle


def main():
    raw_input = np.load("embeddings_real/raw_input.npy").astype(np.float32)[:1, :16, :]
    D = raw_input.shape[-1]
    print("shape:", raw_input.shape, flush=True)

    # Baseline eff_rank only (fast)
    baseline = np.load("embeddings_real/softmax/layer_7.npy").astype(np.float32)
    baseline_flat = baseline.reshape(-1, baseline.shape[-1])
    base_er = effective_rank(baseline_flat)
    print("baseline eff_rank:", base_er, flush=True)

    results = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print("running lam=", lam, "...", flush=True)
        t0 = time.time()
        attn = HybridAttention(d_model=D, num_heads=2, epsilon=0.1, lam=lam, chunk_size=4, seed=42)
        out, A = attn(raw_input, return_attention=True)
        er = effective_rank(out)
        idim = intrinsic_dimension_mle(out)
        rr = er / base_er
        print("  lam=", lam, "eff_rank=", er, "intr_dim=", idim, "rank_ratio=", rr,
              "time=", time.time()-t0, flush=True)
        results.append({"lam": lam, "rank_ratio": rr,
                        "eff_rank": er, "intrinsic_dim_mle": idim})

    Path("results_real").mkdir(exist_ok=True)
    with open("results_real/lambda_sweep_real.json", "w") as f:
        json.dump({
            "mode": "real (Qwen3-0.6B)",
            "baseline_layer_7_eff_rank": base_er,
            "lambda_sweep_results": results,
        }, f, indent=2, default=str)
    print("Saved to results_real/lambda_sweep_real.json", flush=True)


if __name__ == "__main__":
    main()