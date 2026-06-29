"""Final perplexity benchmark: comprehensive comparison.

Configurations:
  1. Baseline                  - standard softmax
  2. Hybrid (1 layer)          - swap only L03 (best per-layer)
  3. Hybrid (mid-layers 3-15)  - swap L03, L07, L11, L15
  4. Hybrid (deep-layers 19-27)- swap L19, L23, L27
  5. Hybrid (all-safe 3-27)    - swap all but L00 and L01
  6. Hybrid (lambda sweep)     - mid-layers with lam in {0.0, 0.25, 0.5, 0.75, 1.0}
"""

import os, sys, json, time, math, argparse
from pathlib import Path

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from perplexity_benchmark_hybrid import (
    load_wikitext_test_text, eval_perplexity,
    swap_attention_layers, restore_attention_layers,
)


def run_config(model, input_ids, label, layers, lam, eps, window, stride, results):
    t0 = time.time()
    original = swap_attention_layers(model, lam=lam, epsilon=eps, layer_indices=layers)
    try:
        ppl = eval_perplexity(model, input_ids, window, stride)
    except Exception as e:
        print(f"  ERROR: {e}")
        ppl = float("nan")
    restore_attention_layers(model, original)
    dt = time.time() - t0
    print(f"  {label}: PPL={ppl:.3f}  ({dt:.1f}s)")
    results[label] = {"layers": layers, "lambda": lam, "eps": eps,
                      "ppl": ppl, "time_s": dt}
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=50_000)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--output-dir", default="results_real/perplexity_final")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINAL PERPLEXITY BENCHMARK: Hybrid Attention vs Baseline")
    print("=" * 70)
    print(f"Dataset: WikiText-2 test, max_chars={args.max_chars}, window={args.window}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base", torch_dtype=torch.float16,
        device_map="cuda", attn_implementation="eager",
    )
    model.eval()

    text = load_wikitext_test_text(max_chars=args.max_chars)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(f"Tokens: {input_ids.shape[1]}, Windows: {(input_ids.shape[1] - args.window) // args.stride}")

    results = {}

    # Baseline
    print("\n[1] Baseline (standard softmax)...")
    t0 = time.time()
    ppl_base = eval_perplexity(model, input_ids, args.window, args.stride)
    base_time = time.time() - t0
    print(f"  Baseline: PPL={ppl_base:.3f}  ({base_time:.1f}s)")
    results["baseline"] = {"layers": None, "lambda": None, "eps": None,
                            "ppl": ppl_base, "time_s": base_time}

    EPS = 0.1
    LAM = 1.0  # pure DeltaNet (best single-layer)

    # Single layer (best per-layer from previous run)
    print(f"\n[2] Single layer swap (L03, lambda={LAM})...")
    run_config(model, input_ids, "single_L03", [3], LAM, EPS, args.window, args.stride, results)

    # Mid layers
    print(f"\n[3] Mid-layers swap (L03-L15, lambda={LAM})...")
    run_config(model, input_ids, "mid_L03-L15", [3, 7, 11, 15], LAM, EPS, args.window, args.stride, results)

    # Deep layers
    print(f"\n[4] Deep-layers swap (L19-L27, lambda={LAM})...")
    run_config(model, input_ids, "deep_L19-L27", [19, 23, 27], LAM, EPS, args.window, args.stride, results)

    # All safe layers (skip L00, L01)
    print(f"\n[5] All-safe layers (L03-L27, lambda={LAM})...")
    run_config(model, input_ids, "safe_L03-L27", list(range(3, 28)), LAM, EPS, args.window, args.stride, results)

    # Lambda sweep on mid-layers
    print("\n[6] Lambda sweep on mid-layers (L03-L15)...")
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        run_config(model, input_ids, f"mid_L03-L15_lam{lam}", [3, 7, 11, 15],
                   lam, EPS, args.window, args.stride, results)

    out_file = out_dir / "ppl_final.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Config':<30} {'PPL':>10} {'dPPL':>10} {'Layers':>15}")
    print("-" * 70)
    base_ppl = results["baseline"]["ppl"]
    for k, r in results.items():
        ppl = r["ppl"]
        d = ppl - base_ppl if ppl == ppl else float("nan")
        layers = "all" if r["layers"] is None else str(r["layers"])
        ppl_str = f"{ppl:.3f}" if ppl == ppl else "NaN"
        d_str = f"{d:+.3f}" if d == d else "NaN"
        print(f"{k:<30} {ppl_str:>10} {d_str:>10} {layers:>15}")
    print("=" * 70)


if __name__ == "__main__":
    main()