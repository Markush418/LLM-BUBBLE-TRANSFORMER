"""
BT V5 Perplexity Evaluation - Delta PPL Gate Test
==================================================
Following BT-V5_05_protocol_positioning.md Sec.1:
  "DeltaPPL = PPL_BT - PPL_softmax <= 2%"

Tests pure SIRI (no DeltaNet, no Power Diagram) on Qwen3-0.6B
with WikiText-2 test split.

Configurations tested:
  1. Baseline (standard softmax)
  2. Pure SIRI across different epsilon values
  3. Pure SIRI on specific layer subsets (safe layers vs all layers)
  4. Lambda sweep (DeltaNet + SIRI interpolation)

Gate: DeltaPPL <= 2% -> proceed; >2% -> diagnose.

Usage:
    py experiments/perplexity_bt_v5_eval.py
    py experiments/perplexity_bt_v5_eval.py --max-chars 50000
    py experiments/perplexity_bt_v5_eval.py --skip-lambda-sweep
"""

import os
import sys
import json
import math
import time
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

from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper


# =============================================================================
# Configuration
# =============================================================================
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
SEED = 42

# BT V5 Doc 05 Sec.2: epsilon values to test
EPSILON_SWEEP = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

# BT V5 Doc 01 Sec.7: safe layers (empirically validated)
SAFE_LAYERS = [3, 7, 11, 15, 19, 23]

# BT V5 Doc 01 Sec.9: hybrid schedule (3:1 ratio)
HYBRID_LAYERS_EVERY_4 = [0, 1, 2, 4, 5, 6, 8, 9, 10,
                          12, 13, 14, 16, 17, 18, 20, 21, 22,
                          24, 25, 26]

# PPL evaluation params
WINDOW = 256
STRIDE = 256


# =============================================================================
# Data loader
# =============================================================================
def load_wikitext_test_text(max_chars=None):
    """Load WikiText-2 test split as a single concatenated string."""
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars is not None:
        text = text[:max_chars]
    return text


# =============================================================================
# PPL evaluation (standard LM eval)
# =============================================================================
@torch.no_grad()
def eval_perplexity(model, input_ids, window=WINDOW, stride=STRIDE, device="cuda"):
    """Sliding-window perplexity: exp(mean(NLL))."""
    n_tokens = input_ids.shape[1]
    nlls = []
    n_tokens_counted = 0

    for begin_loc in range(0, n_tokens - window, stride):
        end_loc = begin_loc + window
        target_ids = input_ids[:, begin_loc:end_loc].to(device)

        outputs = model(target_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = target_ids[:, 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        nlls.append(loss.item())
        n_tokens_counted += shift_labels.numel()

    avg_nll = sum(nlls) / n_tokens_counted
    return math.exp(avg_nll)


# =============================================================================
# Attention swap utilities
# =============================================================================
def swap_attention_siri(model, epsilon, layer_indices=None):
    """Swap layers to pure SIRI (use_delta=False, no Power Diagram).

    Returns list of (layer_idx, original_self_attn) for restoration.
    """
    if layer_indices is None:
        layer_indices = range(len(model.model.layers))

    original_self_attns = []
    for i in layer_indices:
        layer = model.model.layers[i]
        orig_attn = layer.self_attn
        wrapper = Qwen3HybridGQABubbleWrapper(
            original_attn=orig_attn,
            epsilon=epsilon,
            lam=0.0,           # pure SIRI (no DeltaNet contribution)
            use_delta=False,   # disable DeltaNet entirely
            use_psi=False,     # no Power Diagram bias
            siri_mode="classical",
        ).cuda()
        layer.self_attn = wrapper
        original_self_attns.append((i, orig_attn))
    return original_self_attns


def swap_attention_hybrid(model, epsilon, lam, layer_indices=None):
    """Swap layers to Hybrid (DeltaNet + SIRI + psi).

    Returns list of (layer_idx, original_self_attn) for restoration.
    """
    if layer_indices is None:
        layer_indices = range(len(model.model.layers))

    original_self_attns = []
    for i in layer_indices:
        layer = model.model.layers[i]
        orig_attn = layer.self_attn
        wrapper = Qwen3HybridGQABubbleWrapper(
            original_attn=orig_attn,
            epsilon=epsilon,
            lam=lam,
            use_delta=True,
            use_psi=True,
            siri_mode="classical",
        ).cuda()
        layer.self_attn = wrapper
        original_self_attns.append((i, orig_attn))
    return original_self_attns


def restore_attention_layers(model, original_self_attns):
    for i, orig_attn in original_self_attns:
        model.model.layers[i].self_attn = orig_attn


# =============================================================================
# Main evaluation
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="BT V5 Perplexity Gate Test")
    parser.add_argument("--max-chars", type=int, default=200_000,
                        help="Max chars from WikiText-2 (default: 200k)")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=EPSILON_SWEEP,
                        help="Epsilon values to test")
    parser.add_argument("--safe-layers", type=int, nargs="+",
                        default=SAFE_LAYERS,
                        help="Safe layer indices for SIRI")
    parser.add_argument("--skip-lambda-sweep", action="store_true",
                        help="Skip hybrid lambda sweep")
    parser.add_argument("--output-dir", default="results_real/perplexity_bt_v5")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  BT V5 PERPLEXITY GATE TEST")
    print("  Following BT-V5_05_protocol_positioning.md Sec.1")
    print("=" * 70)
    print(f"  Model: {MODEL_ID}")
    print(f"  Dataset: WikiText-2 (max {args.max_chars} chars)")
    print(f"  Epsilon sweep: {args.epsilons}")
    print(f"  Safe layers: {args.safe_layers}")
    print(f"  Window: {args.window}, Stride: {args.stride}")
    print(f"  Gate: Delta PPL <= 2%")
    print("=" * 70)

    # ---- Load model ----
    print("\n[1/5] Loading Qwen3-0.6B-Base...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    print(f"      VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"      Layers: {len(model.model.layers)}")

    # ---- Load data ----
    print(f"\n[2/5] Loading WikiText-2 test (max {args.max_chars} chars)...")
    text = load_wikitext_test_text(max_chars=args.max_chars)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    n_tokens = input_ids.shape[1]
    n_windows = (n_tokens - args.window) // args.stride
    print(f"      {n_tokens} tokens, {n_windows} evaluation windows")

    results = {}

    # ==== PHASE 1: Baseline ====
    print("\n" + "=" * 70)
    print("  PHASE 1: BASELINE (standard softmax)")
    print("=" * 70)
    t0 = time.time()
    ppl_baseline = eval_perplexity(model, input_ids, args.window, args.stride)
    dt = time.time() - t0
    print(f"  PPL = {ppl_baseline:.3f}  ({dt:.1f}s)")
    results["baseline"] = {"ppl": ppl_baseline, "time_s": dt}

    # ==== PHASE 2: Pure SIRI - epsilon sweep (safe layers) ====
    print("\n" + "=" * 70)
    print("  PHASE 2: PURE SIRI - Epsilon Sweep (safe layers)")
    print(f"  Layers: {args.safe_layers}")
    print("=" * 70)

    for eps in args.epsilons:
        key = f"siri_eps{eps}_safe"
        print(f"  -> epsilon={eps}...", end=" ", flush=True)
        original = swap_attention_siri(model, eps, layer_indices=args.safe_layers)
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids, args.window, args.stride)
            delta_pct = (ppl - ppl_baseline) / ppl_baseline * 100
            gate = "PASS" if delta_pct <= 2.0 else "FAIL"
            dt = time.time() - t0
            print(f"PPL={ppl:.3f} (Delta={delta_pct:+.2f}%) [{gate}] ({dt:.1f}s)")
        except Exception as e:
            ppl = float('nan')
            delta_pct = float('nan')
            gate = "ERROR"
            dt = time.time() - t0
            print(f"ERROR: {e} ({dt:.1f}s)")
        restore_attention_layers(model, original)
        results[key] = {
            "ppl": ppl, "delta_pct": delta_pct, "epsilon": eps,
            "layers": args.safe_layers, "gate": gate, "time_s": dt,
        }

    # ==== PHASE 3: Pure SIRI - epsilon sweep (ALL layers) ====
    print("\n" + "=" * 70)
    print("  PHASE 3: PURE SIRI - Epsilon Sweep (ALL 28 layers)")
    print("=" * 70)

    for eps in [0.01, 0.05, 0.1]:  # subset for time
        key = f"siri_eps{eps}_all"
        print(f"  -> epsilon={eps}...", end=" ", flush=True)
        original = swap_attention_siri(model, eps, layer_indices=None)
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids, args.window, args.stride)
            delta_pct = (ppl - ppl_baseline) / ppl_baseline * 100
            gate = "PASS" if delta_pct <= 2.0 else "FAIL"
            dt = time.time() - t0
            print(f"PPL={ppl:.3f} (Delta={delta_pct:+.2f}%) [{gate}] ({dt:.1f}s)")
        except Exception as e:
            ppl = float('nan')
            delta_pct = float('nan')
            gate = "ERROR"
            dt = time.time() - t0
            print(f"ERROR: {e} ({dt:.1f}s)")
        restore_attention_layers(model, original)
        results[key] = {
            "ppl": ppl, "delta_pct": delta_pct, "epsilon": eps,
            "layers": "ALL", "gate": gate, "time_s": dt,
        }

    # ==== PHASE 4: Hybrid Lambda Sweep (safe layers) ====
    if not args.skip_lambda_sweep:
        print("\n" + "=" * 70)
        print("  PHASE 4: HYBRID LAMBDA SWEEP (DeltaNet + SIRI)")
        print(f"  Layers: {args.safe_layers}, Epsilon: 0.01")
        print("=" * 70)

        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            key = f"hybrid_lam{lam}_eps0.01_safe"
            print(f"  -> lambda={lam}...", end=" ", flush=True)
            original = swap_attention_hybrid(
                model, epsilon=0.01, lam=lam, layer_indices=args.safe_layers
            )
            t0 = time.time()
            try:
                ppl = eval_perplexity(model, input_ids, args.window, args.stride)
                delta_pct = (ppl - ppl_baseline) / ppl_baseline * 100
                gate = "PASS" if delta_pct <= 2.0 else "FAIL"
                dt = time.time() - t0
                print(f"PPL={ppl:.3f} (Delta={delta_pct:+.2f}%) [{gate}] ({dt:.1f}s)")
            except Exception as e:
                ppl = float('nan')
                delta_pct = float('nan')
                gate = "ERROR"
                dt = time.time() - t0
                print(f"ERROR: {e} ({dt:.1f}s)")
            restore_attention_layers(model, original)
            results[key] = {
                "ppl": ppl, "delta_pct": delta_pct, "lambda": lam,
                "epsilon": 0.01, "layers": args.safe_layers,
                "gate": gate, "time_s": dt,
            }

    # ==== PHASE 5: Single-layer sensitivity ====
    print("\n" + "=" * 70)
    print("  PHASE 5: SINGLE-LAYER SENSITIVITY (epsilon=0.01)")
    print("=" * 70)

    for layer_idx in range(28):
        key = f"single_layer_{layer_idx}_eps0.01"
        print(f"  -> layer {layer_idx}...", end=" ", flush=True)
        original = swap_attention_siri(model, epsilon=0.01, layer_indices=[layer_idx])
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids, args.window, args.stride)
            delta_pct = (ppl - ppl_baseline) / ppl_baseline * 100
            gate = "PASS" if delta_pct <= 2.0 else "FAIL"
            dt = time.time() - t0
            print(f"PPL={ppl:.3f} (Delta={delta_pct:+.2f}%) [{gate}] ({dt:.1f}s)")
        except Exception as e:
            ppl = float('nan')
            delta_pct = float('nan')
            gate = "ERROR"
            dt = time.time() - t0
            print(f"ERROR: {e} ({dt:.1f}s)")
        restore_attention_layers(model, original)
        results[key] = {
            "ppl": ppl, "delta_pct": delta_pct, "layer": layer_idx,
            "epsilon": 0.01, "gate": gate, "time_s": dt,
        }

    # ==== Save results ====
    out_file = out_dir / "ppl_bt_v5_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # ==== Summary table ====
    print("\n" + "=" * 70)
    print("  SUMMARY - BT V5 Perplexity Gate Test")
    print("=" * 70)
    print(f"  Baseline PPL: {ppl_baseline:.3f}")
    print(f"  Gate threshold: Delta PPL <= 2%")
    print("-" * 70)
    print(f"  {'config':<35} {'PPL':>10} {'Delta%':>8} {'gate':>6}")
    print("-" * 70)

    # Sort by delta
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if k != "baseline"],
        key=lambda x: x[1].get("delta_pct", 999) if x[1].get("delta_pct") == x[1].get("delta_pct") else 999
    )

    for key, r in sorted_results:
        ppl_str = f"{r['ppl']:.3f}" if r['ppl'] == r['ppl'] else "NaN"
        delta_str = f"{r['delta_pct']:+.2f}%" if r['delta_pct'] == r['delta_pct'] else "NaN"
        gate_str = r.get('gate', '?')
        print(f"  {key:<35} {ppl_str:>10} {delta_str:>8} {gate_str:>6}")

    print("=" * 70)

    # ==== Gate verdict ====
    print("\n" + "=" * 70)
    print("  GATE VERDICT")
    print("=" * 70)
    passing = [k for k, v in results.items()
               if v.get("gate") == "PASS" and k != "baseline"]
    failing = [k for k, v in results.items()
               if v.get("gate") == "FAIL" and k != "baseline"]
    print(f"  Passing: {len(passing)}")
    print(f"  Failing: {len(failing)}")
    if failing:
        print(f"  Failing configs: {failing}")
    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
