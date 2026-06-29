"""Layer-stratified perplexity analysis: which layers are most robust to Hybrid replacement?"""
import os, sys, json, time, math
import argparse
from pathlib import Path

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from perplexity_benchmark_hybrid import (
    load_wikitext_test_text, eval_perplexity,
    swap_attention_layers, restore_attention_layers, EPSILON,
)
from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=50_000)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--siri-mode", type=str, default="classical",
                        help="SIRI variant: classical/chiller/sparse/soft")
    parser.add_argument("--output-dir", default="results_real/perplexity_layerwise")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base", torch_dtype=torch.float16,
        device_map="cuda", attn_implementation="eager",
    )
    model.eval()

    text = load_wikitext_test_text(max_chars=args.max_chars)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    n_tokens = input_ids.shape[1]
    n_windows = (n_tokens - args.window) // args.stride
    print(f"Tokens: {n_tokens}, Windows: {n_windows}")

    # Baseline
    print("Baseline...")
    t0 = time.time()
    ppl_base = eval_perplexity(model, input_ids, args.window, args.stride)
    base_time = time.time() - t0
    print(f"  Baseline PPL = {ppl_base:.3f}  ({base_time:.1f}s)")

    results = {"baseline": {"ppl": ppl_base, "time_s": base_time}}
    n_layers = len(model.model.layers)

    # Per-layer swap (one layer at a time, all lambdas)
    print(f"\nPer-layer swap (lambdas={args.lambdas})...")
    layer_subset = [0, 3, 7, 11, 15, 19, 23, 27]  # representative subset
    for layer_idx in layer_subset:
        for lam in args.lambdas:
            key = f"L{layer_idx:02d}_lam{lam}"
            t0 = time.time()
            original = swap_attention_layers(
                model, lam=lam, epsilon=EPSILON, layer_indices=[layer_idx],
                siri_mode=args.siri_mode,
            )
            try:
                ppl = eval_perplexity(model, input_ids, args.window, args.stride)
            except Exception as e:
                print(f"  ERROR layer={layer_idx} lam={lam}: {e}")
                ppl = float("nan")
            restore_attention_layers(model, original)
            dt = time.time() - t0
            print(f"  L{layer_idx:02d} lam={lam:.2f}: PPL={ppl:.3f}  ({dt:.1f}s)  dPPL={ppl - ppl_base:+.3f}")
            results[key] = {"layer": layer_idx, "lambda": lam, "siri_mode": args.siri_mode,
                            "ppl": ppl, "delta_ppl": ppl - ppl_base, "time_s": dt}

    out_file = out_dir / "ppl_per_layer.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Summary: best layer per lambda
    print("\n" + "=" * 60)
    print("BEST PER-LAYER RESULTS")
    print("=" * 60)
    print(f"Baseline: {ppl_base:.3f}")
    for lam in args.lambdas:
        lams_results = [(k, r) for k, r in results.items() if k.startswith("L") and r.get("lambda") == lam]
        best = min(lams_results, key=lambda x: x[1]["ppl"] if x[1]["ppl"] == x[1]["ppl"] else float("inf"))
        print(f"  lam={lam:.2f}: best layer = L{best[1]['layer']:02d}, PPL={best[1]['ppl']:.3f}, dPPL={best[1]['delta_ppl']:+.3f}")


if __name__ == "__main__":
    main()