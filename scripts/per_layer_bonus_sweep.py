"""
Per-Layer Bonus Tuning Sweep
=============================
Strategy: layers with HIGH entropy ratio (good clustering) get higher bonus,
          layers with LOW entropy ratio (collapsed) get lower bonus.

Rule: bonus[layer] = base_bonus * entropy_ratio[layer] / avg_entropy_ratio

This normalizes so the "average" layer gets base_bonus,
well-clustered layers get more, collapsed layers get less.

Uses sliding-window PPL computation (avoids OOM on 4GB VRAM).

Usage:
    python scripts/per_layer_bonus_sweep.py [--base-bonus 0.2 0.3 0.5]
"""

import sys
import json
import time
import math
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

REPO_PATH = Path(r"C:\Users\negocio\Desktop\LLM-BUBBLE")
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
NUM_BUBBLES = 32
TOP_K = 64
EPS_STAR = 0.005
MAX_CHARS = 40000
MAX_LENGTH = 512
STRIDE = 256

OUTPUT_FILE = Path("per_layer_bonus_sweep_results.json")


def compute_perplexity(model, tokenizer, text, desc="PPL"):
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f" Tokens: {seq_len:,} | windows: {MAX_LENGTH} | stride: {STRIDE}")

    nlls = []
    total_toks = 0
    prev_end = 0

    pbar = tqdm(range(0, seq_len, STRIDE), desc=desc, unit="win", dynamic_ncols=True)
    with torch.no_grad():
        for begin in pbar:
            end = min(begin + MAX_LENGTH, seq_len)
            trg_len = end - prev_end
            inp_ids = encodings.input_ids[:, begin:end].to(DEVICE)
            labels = inp_ids.clone()
            labels[:, :-trg_len] = -100
            out = model(inp_ids, labels=labels)
            nlls.append(out.loss.float() * trg_len)
            total_toks += trg_len
            prev_end = end
            pbar.set_postfix({"running_ppl": f"{math.exp(sum(nlls).item() / total_toks):.2f}"})
            if end == seq_len:
                break

    return math.exp(sum(nlls).item() / total_toks)


def swap_all_layers(model, routing_bonus, eps=EPS_STAR):
    from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper
    n_swapped = 0
    for layer_idx, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn
        bonus = routing_bonus if isinstance(routing_bonus, (int, float)) else routing_bonus[layer_idx]
        wrapper = Qwen3GQABubbleWrapper(
            original_attn=original_attn,
            num_bubbles=NUM_BUBBLES,
            top_k=TOP_K,
            eps=eps,
            routing_bonus=bonus,
            debug=(n_swapped == 0),
        )
        layer.self_attn = wrapper
        n_swapped += 1
    return n_swapped


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bonus", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    args = parser.parse_args()

    # Load entropy ratios from analysis
    analysis_file = REPO_PATH / "bubble_assignment_analysis.json"
    if analysis_file.exists():
        with open(analysis_file) as f:
            analysis = json.load(f)
        entropy_ratios = {r["layer"]: r["entropy_ratio"] for r in analysis["per_layer"]}
        avg_er = analysis["summary"]["avg_entropy_ratio"]
        print(f"Loaded entropy ratios from analysis (avg={avg_er:.3f})")
    else:
        print("ERROR: Run analyze_bubble_assignments.py first!")
        return

    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())[:MAX_CHARS]

    # Baseline
    print("\n-- Baseline --")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
    model.eval()
    t0 = time.time()
    ppl_baseline = compute_perplexity(model, tokenizer, text, desc="Baseline")
    t_base = time.time() - t0
    print(f"Baseline PPL: {ppl_baseline:.4f} ({t_base:.1f}s)")

    del model
    torch.cuda.empty_cache()

    n_layers = 28  # Qwen3-0.6B
    results = {"baseline_ppl": ppl_baseline, "baseline_time_s": t_base, "sweeps": []}

    for base_bonus in args.base_bonus:
        # Compute per-layer bonuses
        per_layer_bonuses = []
        for i in range(n_layers):
            er = entropy_ratios.get(i, avg_er)
            bonus = base_bonus * (er / avg_er)
            bonus = max(0.01, min(bonus, 2.0))
            per_layer_bonuses.append(round(bonus, 4))

        print(f"\n-- base_bonus={base_bonus} (per-layer range: [{min(per_layer_bonuses):.4f}, {max(per_layer_bonuses):.4f}]) --")

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
        n_swapped = swap_all_layers(model, per_layer_bonuses)
        print(f" {n_swapped} layers wrapped")

        model.eval()
        t0 = time.time()
        ppl = compute_perplexity(model, tokenizer, text, desc=f"Entropy-tuned b={base_bonus}")
        dt = time.time() - t0

        delta_ppl = ppl - ppl_baseline
        delta_pct = (delta_ppl / ppl_baseline) * 100

        print(f" PPL: {ppl:.4f} (delta={delta_ppl:+.4f}, {delta_pct:+.2f}%) [{dt:.1f}s]")

        results["sweeps"].append({
            "base_bonus": base_bonus,
            "strategy": "entropy_weighted",
            "ppl": ppl,
            "delta_ppl": delta_ppl,
            "delta_pct": delta_pct,
            "time_s": dt,
            "per_layer_bonuses": per_layer_bonuses,
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f" PER-LAYER BONUS TUNING RESULTS")
    print(f"{'='*60}")
    print(f" Baseline PPL: {ppl_baseline:.4f}")
    print(f" {'BaseBonus':>10} {'PPL':>10} {'Delta%':>10}  vs Uniform")
    print(f"{'-'*60}")

    uniform = {0.1: 1.04, 0.2: 2.07, 0.3: 3.04, 0.5: 3.67}
    for s in results["sweeps"]:
        bb = s["base_bonus"]
        uni = uniform.get(bb, "?")
        better = "BETTER" if abs(s["delta_pct"]) < uni else "worse"
        print(f" {bb:>10.2f} {s['ppl']:>10.4f} {s['delta_pct']:>+9.2f}%  vs uniform {uni:+.2f}% [{better}]")
    print(f"{'='*60}")

    results["uniform_comparison"] = uniform
    results["config"] = {
        "model": MODEL_ID,
        "num_bubbles": NUM_BUBBLES,
        "dataset": "wikitext-2-raw-v1",
        "max_chars": MAX_CHARS,
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "tuning_strategy": "bonus[layer] = base * entropy_ratio[layer] / avg_entropy_ratio",
        "entropy_source": str(analysis_file),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
