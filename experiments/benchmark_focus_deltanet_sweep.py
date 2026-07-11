"""
Focus DeltaNet Sweep — Verify NaN fix and find best lambda
==========================================================
Tests FocusBubbleDeltaNet (Focus + DeltaNet) with safe normalization.
Sweeps lambda (interpolation weight) on best layer (L7).
"""

import os, sys, json, math, time
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
EPSILON = 0.001
TAU_ITERS = 1


def load_wikitext(split, max_chars=None):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars:
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


def main():
    out_dir = Path("results_real/focus_bubble")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FOCUS DELTANET SWEEP (post NaN fix)")
    print("=" * 70)

    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    print("\n[2/3] Loading data...")
    text = load_wikitext("test", max_chars=MAX_CHARS)
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    print("\n[3/3] Baseline...")
    ppl_base = eval_perplexity(model, input_ids)
    gate_max = ppl_base * 1.02
    print(f"  PPL_base = {ppl_base:.3f}")
    print(f"  Gate max = {gate_max:.3f}")

    # Lambda sweep on L7 (best layer from previous sweep)
    lambdas = [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    layer = 7

    results = {
        "baseline_ppl": ppl_base,
        "gate_max": gate_max,
        "epsilon": EPSILON,
        "tau_iters": TAU_ITERS,
        "configs": []
    }

    print(f"\nSweeping lambda on L{layer} (eps={EPSILON} tau={TAU_ITERS})...")
    print(f"{'Lambda':>8} {'PPL':>10} {'Delta%':>8} {'Gate':>6} {'NaN':>5}")
    print("-" * 45)

    for lam in lambdas:
        orig_attn = model.model.layers[layer].self_attn
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn, epsilon=EPSILON, tau_iters=TAU_ITERS,
            use_psi=True, use_delta=True, lam=lam,
        ).cuda()
        model.model.layers[layer].self_attn = wrapper

        ppl = eval_perplexity(model, input_ids)
        has_nan = math.isnan(ppl) or math.isinf(ppl)
        delta = (ppl - ppl_base) / ppl_base * 100 if not has_nan else float('nan')
        gate = "PASS" if not has_nan and ppl <= gate_max else "FAIL"

        nan_str = "YES" if has_nan else "no"
        print(f"{lam:8.2f} {ppl:10.3f} {delta:+7.2f}% {gate:>6} {nan_str:>5}")

        results["configs"].append({
            "layer": layer, "epsilon": EPSILON, "tau_iters": TAU_ITERS,
            "lambda": lam, "ppl": ppl, "delta_pct": delta,
            "gate": gate, "nan": has_nan
        })

        model.model.layers[layer].self_attn = orig_attn

    # Summary
    valid = [c for c in results["configs"] if not c["nan"]]
    if valid:
        best = min(valid, key=lambda c: c["ppl"])
        results["best"] = best
    else:
        results["best"] = None

    out_file = out_dir / "focus_deltanet_sweep.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    if best:
        print(f"  BEST: L{best['layer']} eps={best['epsilon']} tau={best['tau_iters']} lambda={best['lambda']}")
        print(f"  PPL={best['ppl']:.3f} ({best['delta_pct']:+.2f}%) [{best['gate']}]")
    else:
        print("  NO VALID CONFIG (all NaN)")
    print(f"{'='*70}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
