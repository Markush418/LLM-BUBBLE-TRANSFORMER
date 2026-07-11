"""
Needle-in-a-Haystack (NIAH) Benchmark — Long-Context Retrieval
================================================================

Tests long-context retrieval by checking if the model assigns higher
logit probability to the needle tokens vs random tokens at various
haystack depths.

Compares:
  1. Softmax baseline (vanilla Qwen3-0.6B-Base)
  2. SIRI attention (epsilon=0.1, siri_mode="soft", first 4 layers)

Uses forward-pass scoring (not generation) for speed on GTX 1650.

Usage:
    py experiments/eval_niah.py
"""

import json
import sys
import time
import gc
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
NEEDLE = "The secret code is XRAY7742"
NEEDLE_TOKENS = ["X", "R", "A", "Y", "7", "7", "4", "2"]
HAYSTACK_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz, judge my vow. "
    "Two driven jocks help fax my big quiz. "
    "The jay, pig, fox, zebra, and my wolves quack! "
    "Sympathizing would fix Quaker problems. "
)
CONTEXT_LENGTHS = [1024, 2048, 4096]
DEPTH_FRAC = [0.25, 0.50, 0.75]
SIRI_WRAP_LAYERS = 4


def build_haystack(target_tokens: int, tokenizer) -> str:
    filler_tokens = tokenizer.encode(HAYSTACK_FILLER, add_special_tokens=False)
    filler_len = max(len(filler_tokens), 1)
    repeats = (target_tokens // filler_len) + 2
    text = HAYSTACK_FILLER * repeats
    tokens = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(tokens)


def insert_needle(haystack: str, depth_frac: float, needle: str) -> str:
    chars = len(haystack)
    insert_pos = int(chars * depth_frac)
    return haystack[:insert_pos] + needle + haystack[insert_pos:]


def make_prompt(text: str) -> str:
    return (
        "Read the following text carefully. Then answer the question.\n\n"
        f"{text}\n\n"
        "Question: What is the secret code? Answer:"
    )


def score_needle_logits(model, tokenizer, prompt: str, needle: str, max_length: int) -> dict:
    """Score by checking logits at the answer position.

    Strategy: tokenize the full prompt, run a forward pass, and check if
    the needle tokens ("XRAY7742") have high logits at the expected
    generation position.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Get logits at the last position (next-token prediction)
    last_logits = logits[0, -1, :]  # [vocab_size]

    # Score: average logit of needle characters
    needle_ids = []
    for ch in needle:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if ids:
            needle_ids.append(ids[0])

    if not needle_ids:
        return {"score": 0.0, "needle_token_ids": [], "needle_logits": [], "top_tokens": []}

    needle_logits = last_logits[needle_ids].float()
    mean_needle_logit = needle_logits.mean().item()

    # Compare to top-k tokens
    topk = torch.topk(last_logits, k=10)
    top_tokens = [tokenizer.decode(tid) for tid in topk.indices]

    # Score: 1.0 if any needle token is in top-10, partial otherwise
    topk_ids = set(topk.indices.tolist())
    hits = sum(1 for tid in needle_ids if tid in topk_ids)
    score = hits / len(needle_ids) if needle_ids else 0.0

    return {
        "score": score,
        "mean_needle_logit": mean_needle_logit,
        "needle_tokens": [tokenizer.decode(tid) for tid in needle_ids],
        "top_10_tokens": top_tokens,
    }


def load_model_and_tokenizer():
    print(f"  Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {MODEL_NAME} (float16, CUDA)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded: {param_count:.1f}M params")
    return model, tokenizer


def wrap_model_siri(model, epsilon: float = 0.1, siri_mode: str = "soft",
                    num_layers: int = SIRI_WRAP_LAYERS):
    print(f"  Wrapping first {num_layers} layers with SIRI (eps={epsilon}, mode={siri_mode})...")
    for i in range(min(num_layers, len(model.model.layers))):
        layer = model.model.layers[i]
        original_attn = layer.self_attn
        wrapper = Qwen3HybridGQABubbleWrapper(
            original_attn=original_attn,
            epsilon=epsilon,
            lam=0.0,
            tau_iters=5,
            use_psi=False,
            use_delta=False,
            siri_mode=siri_mode,
            siri_alpha=0.3,
        ).cuda()
        layer.self_attn = wrapper
    print("  Done.")


def run_niah_single(model, tokenizer, context_length: int, mode_name: str) -> dict:
    max_len = min(context_length + 100, 4096)
    results = {
        "mode": mode_name,
        "context_length": context_length,
        "max_length": max_len,
        "depths": {},
        "mean_score": 0.0,
    }

    for depth in DEPTH_FRAC:
        haystack = build_haystack(context_length, tokenizer)
        text_with_needle = insert_needle(haystack, depth, NEEDLE)
        prompt = make_prompt(text_with_needle)

        t0 = time.time()
        try:
            scoring = score_needle_logits(model, tokenizer, prompt, NEEDLE, max_len)
            s = scoring["score"]
        except Exception as e:
            print(f"    [ERROR] depth={depth}: {e}")
            scoring = {"score": 0.0, "needle_tokens": [], "top_10_tokens": []}
            s = 0.0
        elapsed = time.time() - t0

        results["depths"][str(depth)] = {
            "score": s,
            "mean_needle_logit": scoring.get("mean_needle_logit", 0.0),
            "needle_tokens": scoring.get("needle_tokens", []),
            "top_10_tokens": scoring.get("top_10_tokens", []),
            "time_sec": round(elapsed, 2),
        }
        top10_str = " ".join(scoring.get("top_10_tokens", [])[:5])
        print(f"    depth={depth:.0%}: score={s:.2f}  time={elapsed:.1f}s  top5=[{top10_str}]")

    scores = [d["score"] for d in results["depths"].values()]
    results["mean_score"] = sum(scores) / len(scores) if scores else 0.0
    return results


def run_niah_benchmark(model, tokenizer, mode_name: str) -> list:
    print(f"\n{'='*60}")
    print(f"  NIAH Benchmark — {mode_name}")
    print(f"{'='*60}")
    all_results = []
    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n  Context length: {ctx_len}")
        t_start = time.time()
        try:
            r = run_niah_single(model, tokenizer, ctx_len, mode_name)
            all_results.append(r)
        except Exception as e:
            print(f"  [FATAL] Context {ctx_len}: {e}")
            traceback.print_exc()
            all_results.append({
                "mode": mode_name,
                "context_length": ctx_len,
                "error": str(e),
                "mean_score": 0.0,
            })
        elapsed = time.time() - t_start
        print(f"  Total for ctx={ctx_len}: {elapsed:.1f}s")
        gc.collect()
        torch.cuda.empty_cache()
    return all_results


def main():
    print("=" * 60)
    print("  Needle-in-a-Haystack (NIAH) Benchmark")
    print("  Model: Qwen3-0.6B-Base (float16, CUDA)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. Aborting.")
        sys.exit(1)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    all_mode_results = {}

    # ── Mode 1: Softmax baseline ──────────────────────────────────────────
    print("\n[1/2] Loading softmax baseline...")
    model_sm, tokenizer = load_model_and_tokenizer()
    softmax_results = run_niah_benchmark(model_sm, tokenizer, "softmax")
    all_mode_results["softmax"] = softmax_results
    del model_sm
    gc.collect()
    torch.cuda.empty_cache()

    # ── Mode 2: SIRI attention ────────────────────────────────────────────
    print("\n[2/2] Loading SIRI model...")
    model_siri, tokenizer = load_model_and_tokenizer()
    wrap_model_siri(model_siri, epsilon=0.1, siri_mode="soft")
    siri_results = run_niah_benchmark(model_siri, tokenizer, "siri_eps0.1_soft")
    all_mode_results["siri_eps0.1_soft"] = siri_results
    del model_siri
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for mode, results in all_mode_results.items():
        print(f"\n  Mode: {mode}")
        for r in results:
            if "error" in r:
                print(f"    ctx={r['context_length']:>6}: ERROR — {r['error']}")
            else:
                print(f"    ctx={r['context_length']:>6}: mean_score={r['mean_score']:.2f}")

    output_path = EXPERIMENTS_DIR.parent / "results" / "niah_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_mode_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")
    print("\n[DONE] NIAH benchmark complete.")


if __name__ == "__main__":
    main()
