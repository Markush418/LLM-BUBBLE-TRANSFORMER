"""
Focus Bubble Multi-Layer Swap Benchmark
========================================
Tests multiple Focus Bubble layers simultaneously.
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
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        nlls.append(loss.item())
        n_tokens_counted += shift_labels.numel()
    avg_nll = sum(nlls) / n_tokens_counted
    return math.exp(avg_nll)

def swap_layers(model, layer_indices, epsilon=EPSILON, tau_iters=TAU_ITERS, lam=0.0, use_delta=False):
    """Swap multiple layers with Focus Bubble wrappers. Returns list of original attentions."""
    orig_attns = []
    for idx in layer_indices:
        orig_attn = model.model.layers[idx].self_attn
        orig_attns.append(orig_attn)
        wrapper = Qwen3FocusBubbleWrapper(
            original_attn=orig_attn,
            epsilon=epsilon,
            tau_iters=tau_iters,
            use_psi=True,
            use_delta=use_delta,
            lam=lam,
        ).cuda()
        model.model.layers[idx].self_attn = wrapper
    return orig_attns

def restore_layers(model, layer_indices, orig_attns):
    for idx, orig in zip(layer_indices, orig_attns):
        model.model.layers[idx].self_attn = orig

def main():
    out_dir = Path("results_real/focus_multilayer")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FOCUS BUBBLE MULTI-LAYER SWAP BENCHMARK")
    print("=" * 70)

    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    print("\n[2/4] Loading data...")
    text = load_wikitext("test", max_chars=MAX_CHARS)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(f"  Tokens: {input_ids.shape[1]}")

    print("\n[3/4] Baseline...")
    ppl_base = eval_perplexity(model, input_ids)
    gate_max = ppl_base * 1.02
    print(f"  PPL_base = {ppl_base:.3f}")
    print(f"  Gate max = {gate_max:.3f}")

    # Multi-layer configurations to test
    configs = [
        # (name, layer_indices, use_delta, lam)
        ("Focus_L7", [7], False, 0.0),
        ("Focus_L10", [10], False, 0.0),
        ("Focus_L12", [12], False, 0.0),
        ("Focus_L7+L10", [7, 10], False, 0.0),
        ("Focus_L7+L12", [7, 12], False, 0.0),
        ("Focus_L10+L12", [10, 12], False, 0.0),
        ("Focus_L7+L10+L12", [7, 10, 12], False, 0.0),
        ("FocusDeltaNet_L7", [7], True, 0.3),
        ("FocusDeltaNet_L7+L10", [7, 10], True, 0.3),
        ("FocusDeltaNet_L7+L10+L12", [7, 10, 12], True, 0.3),
    ]

    results = {"baseline_ppl": ppl_base, "gate_max": gate_max, "configs": []}

    print("\n[4/4] Testing multi-layer configs...")
    for name, layers, use_delta, lam in configs:
        print(f"\n  Testing: {name} (layers {layers})")
        t0 = time.time()
        
        orig_attns = swap_layers(model, layers, use_delta=use_delta, lam=lam)
        ppl = eval_perplexity(model, input_ids)
        restore_layers(model, layers, orig_attns)
        
        delta = (ppl - ppl_base) / ppl_base * 100
        gate = "PASS" if ppl <= gate_max else "FAIL"
        dt = time.time() - t0
        
        print(f"    PPL={ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({dt:.1f}s)")
        
        results["configs"].append({
            "name": name, "layers": layers, "use_delta": use_delta, "lam": lam,
            "ppl": ppl, "delta_pct": delta, "gate": gate, "time_s": dt
        })

    # Save
    out_file = out_dir / "multilayer_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for c in results["configs"]:
        print(f"  {c['name']:25s} PPL={c['ppl']:.3f} ({c['delta_pct']:+.2f}%) [{c['gate']}]")
    
    best = min(results["configs"], key=lambda x: x["ppl"])
    print(f"\n  BEST: {best['name']} PPL={best['ppl']:.3f} ({best['delta_pct']:+.2f}%) [{best['gate']}]")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()