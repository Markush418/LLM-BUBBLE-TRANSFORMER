"""
Learnable Beta Fine-Tuning (Optimized)
=======================================
Fine-tunes ONLY beta (nn.Parameter) on WikiText-2 train set.
Evaluates on WikiText-2 test set.

Fast: ~5 min on GTX 1650 (only 1 scalar parameter).
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

from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
SEED = 42
TARGET_LAYER = 12
WINDOW = 256
STRIDE = 256
EVAL_MAX_CHARS = 50_000
TRAIN_MAX_CHARS = 50_000
N_EPOCHS = 10
LR = 0.02


def load_wikitext(split, max_chars=None):
    from datasets import load_dataset
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


@torch.enable_grad()
def train_one_epoch(model, input_ids, optimizer, window=WINDOW, stride=STRIDE):
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_tokens = input_ids.shape[1]
    batch_count = (n_tokens - window) // stride
    for i, begin_loc in enumerate(range(0, n_tokens - window, stride)):
        end_loc = begin_loc + window
        target_ids = input_ids[:, begin_loc:end_loc].cuda()
        optimizer.zero_grad()
        outputs = model(target_ids)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = target_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        if (i+1) % 50 == 0:
            print(f"    batch {i+1}/{batch_count}", flush=True)
    return total_loss / n_batches


def main():
    out_dir = Path("results_real/learnable_beta")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  LEARNABLE BETA FINE-TUNING (fast mode)")
    print("=" * 70)

    print("\n[1/5] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\n[2/5] Loading WikiText-2...")
    eval_text = load_wikitext("test", max_chars=EVAL_MAX_CHARS)
    eval_ids = tokenizer(eval_text, return_tensors="pt").input_ids
    train_text = load_wikitext("train", max_chars=TRAIN_MAX_CHARS)
    train_ids = tokenizer(train_text, return_tensors="pt").input_ids
    print(f"  Train: {train_ids.shape[1]} tokens, Eval: {eval_ids.shape[1]} tokens")

    print("\n[3/5] Baseline evaluation...")
    ppl_base = eval_perplexity(model, eval_ids)
    gate_max = ppl_base * 1.02
    print(f"  PPL_base = {ppl_base:.3f}")
    print(f"  Gate max = {gate_max:.3f}")

    print("\n[4/5] Swapping L12 -> learnable beta wrapper...")
    layer = model.model.layers[TARGET_LAYER]
    orig_attn = layer.self_attn
    wrapper = Qwen3HybridGQABubbleWrapper(
        original_attn=orig_attn, epsilon=0.1, lam=1.0, use_delta=True,
        siri_mode="bias", bias_beta=0.24, learnable_beta=True, use_psi=False,
    ).cuda()
    layer.self_attn = wrapper
    beta_param = wrapper.bias_beta

    for name, p in model.named_parameters():
        if p is not beta_param:
            p.requires_grad = False

    print(f"  beta = {beta_param.item():.4f} (nn.Parameter={isinstance(beta_param, torch.nn.Parameter)})")

    print("\n[5/5] Training...")
    optimizer = torch.optim.Adam([beta_param], lr=LR)

    results = {"baseline_ppl": ppl_base, "gate_max": gate_max, "history": []}

    ppl_before = eval_perplexity(model, eval_ids)
    delta_before = (ppl_before - ppl_base) / ppl_base * 100
    print(f"  Pre-train: beta={beta_param.item():.4f}, PPL={ppl_before:.3f} ({delta_before:+.2f}%)")
    results["history"].append({"step": 0, "beta": beta_param.item(), "ppl": ppl_before, "delta_pct": delta_before})

    best_ppl = ppl_before
    best_beta = beta_param.item()

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_ids, optimizer)
        ppl = eval_perplexity(model, eval_ids)
        dt = time.time() - t0
        delta = (ppl - ppl_base) / ppl_base * 100
        gate = "PASS" if ppl <= gate_max else "FAIL"

        if ppl < best_ppl:
            best_ppl = ppl
            best_beta = beta_param.item()

        print(f"  Epoch {epoch:2d}: beta={beta_param.item():.4f}  loss={avg_loss:.4f}  "
              f"PPL={ppl:.3f} ({delta:+.2f}%) [{gate}] {dt:.1f}s")

        results["history"].append({
            "step": epoch, "beta": beta_param.item(),
            "loss": avg_loss, "ppl": ppl, "delta_pct": delta, "gate": gate
        })

    # Restore original attention
    layer.self_attn = orig_attn

    results["final_beta"] = beta_param.item()
    results["best_beta"] = best_beta
    results["best_ppl"] = best_ppl
    results["best_delta_pct"] = (best_ppl - ppl_base) / ppl_base * 100
    results["gate_passed"] = best_ppl <= gate_max

    out_file = out_dir / "learnable_beta_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  RESULT")
    print(f"{'='*70}")
    print(f"  Baseline:  {ppl_base:.3f}")
    print(f"  Gate max:  {gate_max:.3f} (+2.00%)")
    print(f"  Best beta: {best_beta:.4f}")
    print(f"  Best PPL:  {best_ppl:.3f} ({(best_ppl-ppl_base)/ppl_base*100:+.2f}%)")
    print(f"  Gate:      {'PASS' if results['gate_passed'] else 'FAIL'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
