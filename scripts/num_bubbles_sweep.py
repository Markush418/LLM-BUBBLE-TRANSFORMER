"""
NUM_BUBBLES Sweep
=================
Tests 16, 32, 64 bubbles with entropy-weighted bonus=0.2 (our best config).
Measures how cluster count affects PPL.

Usage:
    python scripts/num_bubbles_sweep.py
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
TOP_K = 64
EPS_STAR = 0.005
MAX_CHARS = 40000
MAX_LENGTH = 512
STRIDE = 256
BASE_BONUS = 0.2
BUBBLE_VALUES = [16, 32, 64]

OUTPUT_FILE = Path("num_bubbles_sweep_results.json")


def compute_perplexity(model, tokenizer, text, desc="PPL"):
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"  Tokens: {seq_len:,} | windows: {MAX_LENGTH} | stride: {STRIDE}")

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


def swap_all_layers(model, per_layer_bonuses, num_bubbles, eps=EPS_STAR):
    from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper
    n_swapped = 0
    for layer_idx, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn
        bonus = per_layer_bonuses[layer_idx]
        wrapper = Qwen3GQABubbleWrapper(
            original_attn=original_attn,
            num_bubbles=num_bubbles,
            top_k=TOP_K,
            eps=eps,
            routing_bonus=bonus,
            debug=(n_swapped == 0),
        )
        layer.self_attn = wrapper
        n_swapped += 1
    return n_swapped


def compute_entropy_weighted_bonuses(base_bonus, num_bubbles):
    if num_bubbles == 32:
        analysis_file = REPO_PATH / "bubble_assignment_analysis.json"
    else:
        analysis_file = None

    if analysis_file and analysis_file.exists():
        with open(analysis_file) as f:
            analysis = json.load(f)
        entropy_ratios = {r["layer"]: r["entropy_ratio"] for r in analysis["per_layer"]}
        avg_er = analysis["summary"]["avg_entropy_ratio"]
    else:
        avg_er = 0.522
        entropy_ratios = {i: avg_er for i in range(28)}

    n_layers = 28
    bonuses = []
    for i in range(n_layers):
        er = entropy_ratios.get(i, avg_er)
        bonus = base_bonus * (er / avg_er)
        bonus = max(0.01, min(bonus, 2.0))
        bonuses.append(round(bonus, 4))
    return bonuses


def main():
    print("=" * 60)
    print(" NUM_BUBBLES SWEEP")
    print(f" Values: {BUBBLE_VALUES}")
    print(f" Tuning: entropy-weighted, base_bonus={BASE_BONUS}")
    print(f" Device: {DEVICE} | dtype: {DTYPE}")
    print("=" * 60)

    print("\nLoading tokenizer and dataset...")
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

    results = {
        "baseline_ppl": ppl_baseline,
        "baseline_time_s": t_base,
        "base_bonus": BASE_BONUS,
        "tuning": "entropy_weighted",
        "sweeps": [],
    }

    for num_bubbles in BUBBLE_VALUES:
        print(f"\n-- NUM_BUBBLES={num_bubbles} --")

        per_layer_bonuses = compute_entropy_weighted_bonuses(BASE_BONUS, num_bubbles)
        print(f"  Bonus range: [{min(per_layer_bonuses):.4f}, {max(per_layer_bonuses):.4f}]")

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
        n_swapped = swap_all_layers(model, per_layer_bonuses, num_bubbles)
        print(f"  {n_swapped} layers wrapped")

        model.eval()
        t0 = time.time()
        ppl = compute_perplexity(model, tokenizer, text, desc=f"B={num_bubbles}")
        dt = time.time() - t0

        delta_ppl = ppl - ppl_baseline
        delta_pct = (delta_ppl / ppl_baseline) * 100

        print(f"  PPL: {ppl:.4f} (delta={delta_ppl:+.4f}, {delta_pct:+.2f}%) [{dt:.1f}s]")

        results["sweeps"].append({
            "num_bubbles": num_bubbles,
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
    print(f" NUM_BUBBLES SWEEP RESULTS")
    print(f"{'='*60}")
    print(f" Baseline PPL: {ppl_baseline:.4f}")
    print(f" {'Bubbles':>10} {'PPL':>10} {'Delta%':>10}")
    print(f"{'-'*60}")
    for s in results["sweeps"]:
        print(f" {s['num_bubbles']:>10} {s['ppl']:>10.4f} {s['delta_pct']:>+9.2f}%")
    print(f"{'='*60}")

    results["config"] = {
        "model": MODEL_ID,
        "base_bonus": BASE_BONUS,
        "tuning": "entropy_weighted",
        "dataset": "wikitext-2-raw-v1",
        "max_chars": MAX_CHARS,
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "eps": EPS_STAR,
        "top_k": TOP_K,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
