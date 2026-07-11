"""
L9 Re-measurement — confirm if +2.28% FAIL replicates or is noise.
Same params: eps=0.001, tau=1, use_psi=True, use_delta=False.
Same seed=42, same input (WikiText-2 test, 50k chars, window=256, stride=256).
"""

import os, sys, math
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
TARGET_LAYER = 9


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
    print("=" * 70)
    print("  L9 RE-MEASUREMENT (eps=0.001, tau=1)")
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

    # L9 measurement
    layer_idx = TARGET_LAYER
    orig_attn = model.model.layers[layer_idx].self_attn
    wrapper = Qwen3FocusBubbleWrapper(
        original_attn=orig_attn, epsilon=EPSILON, tau_iters=TAU_ITERS,
        use_psi=True, use_delta=False, lam=0.0,
    ).cuda()
    model.model.layers[layer_idx].self_attn = wrapper

    ppl = eval_perplexity(model, input_ids)
    delta = (ppl - ppl_base) / ppl_base * 100
    gate = "PASS" if ppl <= gate_max else "FAIL"

    model.model.layers[layer_idx].self_attn = orig_attn

    print(f"\n{'='*70}")
    print(f"  L9 RESULTS")
    print(f"{'='*70}")
    print(f"  L9 corrida original:  PPL=23.026  Delta=+2.28%  [FAIL]")
    print(f"  L9 corrida nueva:     PPL={ppl:.3f}  Delta={delta:+.2f}%  [{gate}]")
    print(f"  Diferencia absoluta:  {abs(ppl - 23.026):.4f} PPL points")
    print(f"  Diferencia delta:     {abs(delta - 2.28):.2f} percentage points")
    print(f"{'='*70}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
