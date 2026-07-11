"""
Fine Beta Sweep: DeltaNet + SIRI Bias
======================================
Narrowing down the optimal beta for Approach 3.
Tests beta in [0.15, 0.18, 0.20, 0.22, 0.25] on L12 and L10.
Also tests L10+L12 combination.

Baseline PPL: ~22.5 (50k chars)
Gate: DeltaPPL <= 2% (max PPL ~22.95)
"""

import os
import sys
import json
import math
import time
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

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
SEED = 42
WINDOW = 256
STRIDE = 256
MAX_CHARS = 50_000


def load_wikitext_test_text(max_chars=None):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars is not None:
        text = text[:max_chars]
    return text


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


def swap_layers(model, wrapper_kwargs, layer_indices):
    originals = []
    for i in layer_indices:
        layer = model.model.layers[i]
        orig = layer.self_attn
        wrapper = Qwen3HybridGQABubbleWrapper(original_attn=orig, **wrapper_kwargs).cuda()
        layer.self_attn = wrapper
        originals.append((i, orig))
    return originals


def restore_layers(model, originals):
    for i, orig in originals:
        model.model.layers[i].self_attn = orig


def main():
    out_dir = Path("results_real/fine_beta_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FINE BETA SWEEP: DeltaNet + SIRI Bias")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    text = load_wikitext_test_text(max_chars=MAX_CHARS)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    n_tokens = input_ids.shape[1]
    n_windows = (n_tokens - WINDOW) // STRIDE
    print(f"  {n_tokens} tokens, {n_windows} windows")

    results = {}

    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\n--- BASELINE ---")
    t0 = time.time()
    ppl_base = eval_perplexity(model, input_ids)
    dt = time.time() - t0
    print(f"  PPL = {ppl_base:.3f} ({dt:.1f}s)")
    results["baseline"] = {"ppl": ppl_base}

    gate_max = ppl_base * 1.02
    print(f"  Gate max PPL: {gate_max:.3f}")

    # =========================================================================
    # PHASE 1: Fine beta sweep on L12
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: Fine beta sweep on L12")
    print("=" * 70)

    betas = [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.25, 0.28, 0.30]
    for beta in betas:
        key = f"L12_beta{beta:.2f}"
        print(f"  -> beta={beta:.2f}...", end=" ", flush=True)
        originals = swap_layers(model, {
            "epsilon": 0.1, "lam": 1.0, "use_delta": True,
            "siri_mode": "bias", "bias_beta": beta, "use_psi": False,
        }, [12])
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if ppl <= gate_max else "FAIL"
            print(f"PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"ERROR: {e}")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "beta": beta, "layer": "L12"}

    # =========================================================================
    # PHASE 2: Fine beta sweep on L10
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: Fine beta sweep on L10")
    print("=" * 70)

    for beta in betas:
        key = f"L10_beta{beta:.2f}"
        print(f"  -> beta={beta:.2f}...", end=" ", flush=True)
        originals = swap_layers(model, {
            "epsilon": 0.1, "lam": 1.0, "use_delta": True,
            "siri_mode": "bias", "bias_beta": beta, "use_psi": False,
        }, [10])
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if ppl <= gate_max else "FAIL"
            print(f"PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl, delta, gate = float('nan'), float('nan'), "ERROR"
            print(f"ERROR: {e}")
        restore_layers(model, originals)
        results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "beta": beta, "layer": "L10"}

    # =========================================================================
    # PHASE 3: Best beta on L10+L12 combination
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: Best beta on L10+L12")
    print("=" * 70)

    # Find best beta from L12 and L10
    def find_best_layer(prefix):
        best_beta, best_ppl = None, float('inf')
        for k, v in results.items():
            if k.startswith(prefix) and v['gate'] != 'ERROR' and v['ppl'] == v['ppl']:
                if v['ppl'] < best_ppl:
                    best_ppl = v['ppl']
                    best_beta = v['beta']
        return best_beta

    best_beta_l12 = find_best_layer("L12_")
    best_beta_l10 = find_best_layer("L10_")

    if best_beta_l12:
        print(f"\n  Best L12 beta: {best_beta_l12}")
        for beta in [best_beta_l12]:
            key = f"L10+L12_beta{beta:.2f}"
            print(f"  -> L10+L12 beta={beta:.2f}...", end=" ", flush=True)
            originals = swap_layers(model, {
                "epsilon": 0.1, "lam": 1.0, "use_delta": True,
                "siri_mode": "bias", "bias_beta": beta, "use_psi": False,
            }, [10, 12])
            t0 = time.time()
            try:
                ppl = eval_perplexity(model, input_ids)
                delta = (ppl - ppl_base) / ppl_base * 100
                gate = "PASS" if ppl <= gate_max else "FAIL"
                print(f"PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
            except Exception as e:
                ppl, delta, gate = float('nan'), float('nan'), "ERROR"
                print(f"ERROR: {e}")
            restore_layers(model, originals)
            results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "beta": beta, "layer": "L10+L12"}

    # =========================================================================
    # PHASE 4: Epsilon sweep with best beta on L12
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PHASE 4: Epsilon sweep with best beta on L12")
    print("=" * 70)

    if best_beta_l12:
        for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
            key = f"L12_beta{best_beta_l12:.2f}_eps{eps}"
            print(f"  -> eps={eps}...", end=" ", flush=True)
            originals = swap_layers(model, {
                "epsilon": eps, "lam": 1.0, "use_delta": True,
                "siri_mode": "bias", "bias_beta": best_beta_l12, "use_psi": False,
            }, [12])
            t0 = time.time()
            try:
                ppl = eval_perplexity(model, input_ids)
                delta = (ppl - ppl_base) / ppl_base * 100
                gate = "PASS" if ppl <= gate_max else "FAIL"
                print(f"PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
            except Exception as e:
                ppl, delta, gate = float('nan'), float('nan'), "ERROR"
                print(f"ERROR: {e}")
            restore_layers(model, originals)
            results[key] = {"ppl": ppl, "delta_pct": delta, "gate": gate, "beta": best_beta_l12, "epsilon": eps, "layer": "L12"}

    # =========================================================================
    # SAVE
    # =========================================================================
    out_file = out_dir / "fine_beta_results.json"
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
    print(f"  Gate: <= {gate_max:.3f} (+2.00%)")
    print("-" * 70)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if k != "baseline"],
        key=lambda x: x[1].get("ppl", 999) if x[1].get("ppl") == x[1].get("ppl") else 999
    )

    print(f"  {'config':<35} {'PPL':>10} {'Delta%':>10} {'gate':>6}")
    print("-" * 70)
    for key, r in sorted_results:
        ppl_str = f"{r['ppl']:.3f}" if r['ppl'] == r['ppl'] else "NaN"
        delta_str = f"{r['delta_pct']:+.2f}%" if r['delta_pct'] == r['delta_pct'] else "NaN"
        gate_str = r.get('gate', '?')
        print(f"  {key:<35} {ppl_str:>10} {delta_str:>10} {gate_str:>6}")

    print("=" * 70)
    passing = [k for k, v in results.items() if v.get("gate") == "PASS"]
    print(f"\n  GATE: {len(passing)} PASS")
    if passing:
        print(f"  PASSING CONFIGS: {passing}")
    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
