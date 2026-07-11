"""
BT V5 Benchmark: 3 Hybrid Approaches
====================================
Tests three new fusion strategies to pass the DeltaPPL <= 2% gate.

Approach 1: Dot-product cost + row-stochastic (no Sinkhorn columns)
Approach 2: Hybrid cost (geometric + dot-product) with alpha param
Approach 3: DeltaNet base + SIRI positional bias

Baseline: Qwen3-0.6B, WikiText-2, PPL = 22.513
Gate: DeltaPPL <= 2% (max PPL ~22.96)

Usage:
    py experiments/benchmark_hybrid_approaches.py
    py experiments/benchmark_hybrid_approaches.py --max-chars 20000
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
import torch.nn.functional as F
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
SAFE_LAYERS = [3, 7, 11, 15, 19, 23]
WINDOW = 256
STRIDE = 256


# =============================================================================
# Data loader
# =============================================================================
def load_wikitext_test_text(max_chars=None):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars is not None:
        text = text[:max_chars]
    return text


# =============================================================================
# PPL evaluation
# =============================================================================
@torch.no_grad()
def eval_perplexity(model, input_ids, window=WINDOW, stride=STRIDE):
    n_tokens = input_ids.shape[1]
    nlls = []
    n_tokens_counted = 0
    for begin_loc in range(0, n_tokens - window, stride):
        end_loc = begin_loc + window
        target_ids = input_ids[:, begin_loc:end_loc].cuda()
        outputs = model(target_ids)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = target_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        nlls.append(loss.item())
        n_tokens_counted += shift_labels.numel()
    avg_nll = sum(nlls) / n_tokens_counted
    return math.exp(avg_nll)


# =============================================================================
# Layer swap utilities
# =============================================================================
def swap_layers(model, wrapper_kwargs, layer_indices):
    originals = []
    for i in layer_indices:
        layer = model.model.layers[i]
        orig = layer.self_attn
        wrapper = Qwen3HybridGQABubbleWrapper(
            original_attn=orig,
            **wrapper_kwargs,
        ).cuda()
        layer.self_attn = wrapper
        originals.append((i, orig))
    return originals


def restore_layers(model, originals):
    for i, orig in originals:
        model.model.layers[i].self_attn = orig


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=50_000)
    parser.add_argument("--output-dir", default="results_real/hybrid_approaches")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  BT V5 BENCHMARK: 3 HYBRID APPROACHES")
    print("=" * 70)

    # Load model
    print("\nLoading Qwen3-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load data
    text = load_wikitext_test_text(max_chars=args.max_chars)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    n_tokens = input_ids.shape[1]
    n_windows = (n_tokens - WINDOW) // STRIDE
    print(f"  {n_tokens} tokens, {n_windows} windows")

    results = {}

    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("  BASELINE (softmax)")
    print("=" * 70)
    t0 = time.time()
    ppl_base = eval_perplexity(model, input_ids)
    dt = time.time() - t0
    print(f"  PPL = {ppl_base:.3f} ({dt:.1f}s)")
    results["baseline"] = {"ppl": ppl_base, "time_s": dt}

    # =========================================================================
    # APPROACH 1: Dot-product + row-stochastic
    # =========================================================================
    print("\n" + "=" * 70)
    print("  APPROACH 1: Dot-product + Row-Stochastic (L12)")
    print("=" * 70)

    for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
        key = f"approach1_dotproduct_eps{eps}"
        print(f"  -> eps={eps}...", end=" ", flush=True)
        originals = swap_layers(model, {
            "epsilon": eps, "lam": 1.0, "use_delta": True,
            "siri_mode": "dotproduct", "use_psi": False,
        }, [12])
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if delta <= 2.0 else "FAIL"
            print(f"PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"ERROR: {e} ({time.time()-t0:.1f}s)")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "epsilon": eps, "layer": 12}

    # =========================================================================
    # APPROACH 2: Hybrid cost (geometric + dot-product)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  APPROACH 2: Hybrid Cost (L12)")
    print("=" * 70)

    # First: alpha sweep with eps=0.1 fixed
    print("\n  Phase A: Alpha sweep (eps=0.1)")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        key = f"approach2_hybrid_alpha{alpha}_eps0.1"
        print(f"  -> alpha={alpha}...", end=" ", flush=True)
        originals = swap_layers(model, {
            "epsilon": 0.1, "lam": 1.0, "use_delta": True,
            "siri_mode": "hybrid", "hybrid_alpha": alpha, "use_psi": False,
        }, [12])
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if delta <= 2.0 else "FAIL"
            print(f"PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"ERROR: {e} ({time.time()-t0:.1f}s)")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "alpha": alpha, "epsilon": 0.1, "layer": 12}

    # Second: epsilon sweep with best alpha (will be determined after Phase A)
    # For now, use alpha=0.5 as default
    print("\n  Phase B: Epsilon sweep (alpha=0.5)")
    for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
        key = f"approach2_hybrid_alpha0.5_eps{eps}"
        print(f"  -> eps={eps}...", end=" ", flush=True)
        originals = swap_layers(model, {
            "epsilon": eps, "lam": 1.0, "use_delta": True,
            "siri_mode": "hybrid", "hybrid_alpha": 0.5, "use_psi": False,
        }, [12])
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if delta <= 2.0 else "FAIL"
            print(f"PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"ERROR: {e} ({time.time()-t0:.1f}s)")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "alpha": 0.5, "epsilon": eps, "layer": 12}

    # =========================================================================
    # APPROACH 3: DeltaNet + SIRI bias
    # =========================================================================
    print("\n" + "=" * 70)
    print("  APPROACH 3: DeltaNet + SIRI Bias (L12)")
    print("=" * 70)

    # Beta sweep with eps=0.1 fixed
    print("\n  Phase A: Beta sweep (eps=0.1)")
    for beta in [0.0, 0.05, 0.1, 0.2, 0.5]:
        key = f"approach3_bias_beta{beta}_eps0.1"
        print(f"  -> beta={beta}...", end=" ", flush=True)
        originals = swap_layers(model, {
            "epsilon": 0.1, "lam": 1.0, "use_delta": True,
            "siri_mode": "bias", "bias_beta": beta, "use_psi": False,
        }, [12])
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if delta <= 2.0 else "FAIL"
            print(f"PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"ERROR: {e} ({time.time()-t0:.1f}s)")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "beta": beta, "epsilon": 0.1, "layer": 12}

    # Epsilon sweep with best beta (default 0.1)
    print("\n  Phase B: Epsilon sweep (beta=0.1)")
    for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
        key = f"approach3_bias_beta0.1_eps{eps}"
        print(f"  -> eps={eps}...", end=" ", flush=True)
        originals = swap_layers(model, {
            "epsilon": eps, "lam": 1.0, "use_delta": True,
            "siri_mode": "bias", "bias_beta": 0.1, "use_psi": False,
        }, [12])
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if delta <= 2.0 else "FAIL"
            print(f"PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"ERROR: {e} ({time.time()-t0:.1f}s)")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "beta": 0.1, "epsilon": eps, "layer": 12}

    # =========================================================================
    # BEST OF EACH: Safe layers
    # =========================================================================
    print("\n" + "=" * 70)
    print("  BEST OF EACH APPROACH: Safe layers [3,7,11,15,19,23]")
    print("=" * 70)

    # Approach 1: best epsilon from L12 results
    # Approach 2: best alpha+epsilon from L12 results
    # Approach 3: best beta+epsilon from L12 results

    # Find best of each approach
    def find_best(prefix):
        best_key, best_delta = None, float('inf')
        for k, v in results.items():
            if k.startswith(prefix) and v['gate'] != 'ERROR' and v['delta_pct'] == v['delta_pct']:
                if abs(v['delta_pct']) < abs(best_delta):
                    best_delta = v['delta_pct']
                    best_key = k
        return best_key, results.get(best_key, {})

    best1_key, best1 = find_best("approach1_")
    best2_key, best2 = find_best("approach2_")
    best3_key, best3 = find_best("approach3_")

    # Approach 1 best on safe layers
    if best1:
        eps = best1.get("epsilon", 0.1)
        key = f"approach1_dotproduct_eps{eps}_safe"
        print(f"\n  Approach 1 (dotproduct, eps={eps}) on safe layers...")
        originals = swap_layers(model, {
            "epsilon": eps, "lam": 1.0, "use_delta": True,
            "siri_mode": "dotproduct", "use_psi": False,
        }, SAFE_LAYERS)
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if delta <= 2.0 else "FAIL"
            print(f"    PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"    ERROR: {e}")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "epsilon": eps, "layers": "safe"}

    # Approach 2 best on safe layers
    if best2:
        alpha = best2.get("alpha", 0.5)
        eps = best2.get("epsilon", 0.1)
        key = f"approach2_hybrid_alpha{alpha}_eps{eps}_safe"
        print(f"  Approach 2 (hybrid, alpha={alpha}, eps={eps}) on safe layers...")
        originals = swap_layers(model, {
            "epsilon": eps, "lam": 1.0, "use_delta": True,
            "siri_mode": "hybrid", "hybrid_alpha": alpha, "use_psi": False,
        }, SAFE_LAYERS)
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if delta <= 2.0 else "FAIL"
            print(f"    PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"    ERROR: {e}")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "alpha": alpha, "epsilon": eps, "layers": "safe"}

    # Approach 3 best on safe layers
    if best3:
        beta = best3.get("beta", 0.1)
        eps = best3.get("epsilon", 0.1)
        key = f"approach3_bias_beta{beta}_eps{eps}_safe"
        print(f"  Approach 3 (bias, beta={beta}, eps={eps}) on safe layers...")
        originals = swap_layers(model, {
            "epsilon": eps, "lam": 1.0, "use_delta": True,
            "siri_mode": "bias", "bias_beta": beta, "use_psi": False,
        }, SAFE_LAYERS)
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if delta <= 2.0 else "FAIL"
            print(f"    PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"    ERROR: {e}")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "beta": beta, "epsilon": eps, "layers": "safe"}

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    out_file = out_dir / "hybrid_approaches_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Baseline PPL: {ppl_base:.3f}")
    print(f"  Gate: DeltaPPL <= 2% (max PPL = {ppl_base * 1.02:.3f})")
    print("-" * 70)
    print(f"  {'config':<45} {'PPL':>10} {'Delta%':>10} {'gate':>6}")
    print("-" * 70)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if k != "baseline"],
        key=lambda x: x[1].get("delta_pct", 999) if x[1].get("delta_pct") == x[1].get("delta_pct") else 999
    )

    for key, r in sorted_results:
        ppl_str = f"{r['ppl']:.3f}" if r['ppl'] == r['ppl'] else "NaN"
        delta_str = f"{r['delta_pct']:+.2f}%" if r['delta_pct'] == r['delta_pct'] else "NaN"
        gate_str = r.get('gate', '?')
        print(f"  {key:<45} {ppl_str:>10} {delta_str:>10} {gate_str:>6}")

    print("=" * 70)

    # Gate verdict
    passing = [k for k, v in results.items() if v.get("gate") == "PASS"]
    print(f"\n  GATE: {len(passing)} PASS, {len(results)-len(passing)-1} FAIL")
    if passing:
        print(f"  PASSING: {passing}")
    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
