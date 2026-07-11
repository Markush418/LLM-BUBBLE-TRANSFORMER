"""
Focus Bubble Benchmark — Evaluate on Qwen3-0.6B / WikiText-2
============================================================

[PROTOCOL] BT-V5_05_protocol_positioning.md Sec. 1
Gate: DeltaPPL <= 2%

Evaluates FocusBubbleAttention as drop-in replacement for softmax attention.
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

from qwen3_focus_bubble_wrapper import Qwen3FocusBubbleWrapper

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
SEED = 42
WINDOW = 256
STRIDE = 256
MAX_CHARS = 50_000


def load_wikitext(split, max_chars=None):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars is not None:
        text = text[:max_chars]
    return text


@torch.no_grad()
def eval_perplexity(model, input_ids, window=WINDOW, stride=STRIDE):
    model.eval()
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


def swap_layer(model, layer_idx, epsilon, tau_iters, use_psi, use_delta, lam):
    """Swap a single layer with FocusBubbleWrapper."""
    layer = model.model.layers[layer_idx]
    orig_attn = layer.self_attn
    wrapper = Qwen3FocusBubbleWrapper(
        original_attn=orig_attn,
        epsilon=epsilon,
        tau_iters=tau_iters,
        use_psi=use_psi,
        use_delta=use_delta,
        lam=lam,
    ).cuda()
    layer.self_attn = wrapper
    return orig_attn


def restore_layer(model, layer_idx, orig_attn):
    """Restore original attention."""
    model.model.layers[layer_idx].self_attn = orig_attn


def main():
    out_dir = Path("results_real/focus_bubble")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FOCUS BUBBLE BENCHMARK")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading Qwen3-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load data
    print("\n[2/4] Loading WikiText-2...")
    text = load_wikitext("test", max_chars=MAX_CHARS)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(f"  Tokens: {input_ids.shape[1]}")

    # Evaluate baseline
    print("\n[3/4] Baseline evaluation...")
    ppl_base = eval_perplexity(model, input_ids)
    gate_max = ppl_base * 1.02
    print(f"  PPL_base = {ppl_base:.3f}")
    print(f"  Gate max = {gate_max:.3f} (+2.00%)")

    # Test configurations
    print("\n[4/4] Testing Focus Bubble configurations...")
    
    configs = [
        # (layer, epsilon, tau_iters, use_psi, use_delta, lam, description)
        (12, 0.1, 5, True, False, 0.0, "L12 FocusOnly psi"),
        (12, 0.1, 5, False, False, 0.0, "L12 FocusOnly no-psi"),
        (12, 0.01, 5, True, False, 0.0, "L12 FocusOnly eps=0.01"),
        (12, 0.5, 5, True, False, 0.0, "L12 FocusOnly eps=0.5"),
        (12, 0.1, 5, True, True, 0.5, "L12 FocusDeltaNet lam=0.5"),
        (12, 0.1, 5, True, True, 0.8, "L12 FocusDeltaNet lam=0.8"),
        (12, 0.1, 5, True, True, 0.9, "L12 FocusDeltaNet lam=0.9"),
        (12, 0.1, 5, True, True, 1.0, "L12 DeltaNetOnly"),
        (10, 0.1, 5, True, False, 0.0, "L10 FocusOnly psi"),
        (10, 0.1, 5, True, True, 0.9, "L10 FocusDeltaNet lam=0.9"),
        (3, 0.1, 5, True, False, 0.0, "L3 FocusOnly psi"),
        (7, 0.1, 5, True, False, 0.0, "L7 FocusOnly psi"),
    ]
    
    results = {
        "baseline_ppl": ppl_base,
        "gate_max": gate_max,
        "configs": [],
    }
    
    for layer_idx, epsilon, tau_iters, use_psi, use_delta, lam, desc in configs:
        print(f"\n  Testing: {desc}")
        t0 = time.time()
        
        orig_attn = swap_layer(model, layer_idx, epsilon, tau_iters, use_psi, use_delta, lam)
        ppl = eval_perplexity(model, input_ids)
        restore_layer(model, layer_idx, orig_attn)
        
        delta = (ppl - ppl_base) / ppl_base * 100
        gate = "PASS" if ppl <= gate_max else "FAIL"
        dt = time.time() - t0
        
        print(f"    PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({dt:.1f}s)")
        
        results["configs"].append({
            "layer": layer_idx,
            "epsilon": epsilon,
            "tau_iters": tau_iters,
            "use_psi": use_psi,
            "use_delta": use_delta,
            "lam": lam,
            "description": desc,
            "ppl": ppl,
            "delta_pct": delta,
            "gate": gate,
            "time_s": dt,
        })
    
    # Find best
    best = min(results["configs"], key=lambda c: c["ppl"])
    results["best"] = best
    
    # Save
    out_file = out_dir / "focus_bubble_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Baseline PPL: {ppl_base:.3f}")
    print(f"  Gate max:     {gate_max:.3f} (+2.00%)")
    print(f"\n  Best: {best['description']}")
    print(f"    PPL:   {best['ppl']:.3f} ({best['delta_pct']:+.2f}%)")
    print(f"    Gate:  {best['gate']}")
    print("=" * 70)
    
    # Full results table
    print("\n  All results:")
    for c in results["configs"]:
        print(f"    {c['description']:40s} PPL={c['ppl']:.3f} ({c['delta_pct']:+.2f}%) [{c['gate']}]")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
