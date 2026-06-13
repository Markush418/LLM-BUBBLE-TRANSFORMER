"""
All-Layer Bonus Sweep — Find <5% PPL degradation sweet spot
=============================================================
Runs baseline once, then tests routing_bonus = 0.1, 0.2, 0.3 across ALL 28 layers.
Saves all results to all_layer_bonus_sweep_results.json

Usage:
    python scripts/all_layer_bonus_sweep.py
"""

import sys
import math
import json
import time
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET = "wikitext"
DATASET_CFG = "wikitext-2-raw-v1"
SPLIT = "test"

MAX_LENGTH = 512
STRIDE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

EPS_STAR = 0.005
SEED = 42
NUM_BUBBLES = 32
TOP_K = 64

BONUS_VALUES = [0.1, 0.2, 0.3]
MAX_CHARS = 40000

OUTPUT_FILE = Path("all_layer_bonus_sweep_results.json")

REPO_PATH = Path(r"C:\Users\negocio\Desktop\LLM-BUBBLE")

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def load_wikitext2_text() -> str:
    print("-> Cargando WikiText-2 test split...")
    dataset = load_dataset(DATASET, DATASET_CFG, split=SPLIT)
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        print(f" [LIMITED] {MAX_CHARS} chars")
    return text


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
            end = min(begin + max_length, seq_len) if (max_length := MAX_LENGTH) else seq_len
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


def swap_all_layers(model, eps, routing_bonus):
    if str(REPO_PATH) not in sys.path:
        sys.path.insert(0, str(REPO_PATH))
    from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper

    n_swapped = 0
    for layer_idx, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn
        wrapper = Qwen3GQABubbleWrapper(
            original_attn=original_attn,
            num_bubbles=NUM_BUBBLES,
            top_k=TOP_K,
            eps=eps,
            routing_bonus=routing_bonus,
            debug=(n_swapped == 0),
        )
        layer.self_attn = wrapper
        n_swapped += 1

    print(f" [OK] {n_swapped} layers wrapped (bonus={routing_bonus})")
    return n_swapped


def main():
    results = {"bonus_sweep": [], "baseline_ppl": None}

    print(f"\n{'='*60}")
    print(f" All-Layer Bonus Sweep - WikiText-2")
    print(f" Device: {DEVICE} | dtype: {DTYPE}")
    print(f" eps: {EPS_STAR} | bubbles: {NUM_BUBBLES} | top_k: {TOP_K}")
    print(f" Bonus values: {BONUS_VALUES}")
    print(f"{'='*60}\n")

    text = load_wikitext2_text()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ── 1. Baseline (run once) ──────────────────────────────────────────
    print("-- [BASELINE] Softmax standard --")
    t0 = time.time()
    model_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
    model_base.eval()
    ppl_baseline = compute_perplexity(model_base, tokenizer, text, desc="Baseline")
    t_base = time.time() - t0
    print(f"\n OK PPL baseline = {ppl_baseline:.4f} ({t_base:.1f}s)\n")

    results["baseline_ppl"] = ppl_baseline
    results["baseline_time_s"] = t_base

    del model_base
    torch.cuda.empty_cache()

    # ── 2. Sweep bonus values ───────────────────────────────────────────
    for bonus in BONUS_VALUES:
        print(f"\n-- [BONUS={bonus}] All 28 layers --")
        t0 = time.time()

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)

        if str(REPO_PATH) not in sys.path:
            sys.path.insert(0, str(REPO_PATH))

        n_swapped = swap_all_layers(model, eps=EPS_STAR, routing_bonus=bonus)
        model.eval()

        ppl = compute_perplexity(model, tokenizer, text, desc=f"Bonus={bonus}")
        t_run = time.time() - t0

        delta = ppl - ppl_baseline
        delta_pct = (delta / ppl_baseline) * 100

        print(f"\n OK PPL bonus={bonus} = {ppl:.4f} (delta={delta:+.4f}, {delta_pct:+.2f}%) ({t_run:.1f}s)\n")

        results["bonus_sweep"].append({
            "bonus": bonus,
            "ppl": ppl,
            "delta_ppl": delta,
            "delta_pct": delta_pct,
            "time_s": t_run,
            "layers_swapped": n_swapped,
        })

        del model
        torch.cuda.empty_cache()

    # ── 3. Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" SUMMARY - Baseline PPL: {ppl_baseline:.4f}")
    print(f"{'-'*60}")
    print(f" {'Bonus':>8} {'PPL':>10} {'Delta':>10} {'Delta%':>10}")
    print(f"{'-'*60}")
    for r in results["bonus_sweep"]:
        print(f" {r['bonus']:>8.1f} {r['ppl']:>10.4f} {r['delta_ppl']:>+10.4f} {r['delta_pct']:>+9.2f}%")
    print(f"{'='*60}\n")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
