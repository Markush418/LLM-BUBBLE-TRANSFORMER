"""
Geometric comparison L9-L12: concentration_ratio and pairwise L2_diff.
Uses non-triviality verification method (same as verify_focus_nontrivial.py).
Compares FocusBubble attention matrices across 4 layers, not vs baseline.
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
MAX_CHARS = 50_000
EPSILON = 0.001
TAU_ITERS = 1
LAYERS = [9, 10, 11, 12]


def load_wikitext(split, max_chars=None):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars:
        text = text[:max_chars]
    return text


def concentration_ratio(A_np, threshold=1e-3):
    total = A_np.size
    above = (A_np > threshold).sum()
    return above / total


@torch.no_grad()
def extract_attention(model, tokenizer, target_ids, layer_idx, eps, tau):
    """Extract [B, H, N, N] attention matrix for a given layer."""
    # Swap layer
    layer = model.model.layers[layer_idx]
    orig_attn = layer.self_attn
    wrapper = Qwen3FocusBubbleWrapper(
        original_attn=orig_attn, epsilon=eps, tau_iters=tau,
        use_psi=True, use_delta=False, lam=0.0,
    ).cuda()
    layer.self_attn = wrapper

    # Forward with output_attentions
    outputs = model(target_ids, output_attentions=True)
    A = outputs.attentions[layer_idx]  # [B, H, N, N]

    # Restore
    layer.self_attn = orig_attn
    return A


@torch.no_grad()
def extract_softmax_attention(model, target_ids, layer_idx):
    """Extract softmax [B, H, N, N] attention matrix for a given layer."""
    outputs = model(target_ids, output_attentions=True)
    A = outputs.attentions[layer_idx]  # [B, H, N, N]
    return A


def main():
    print("=" * 70)
    print("  GEOMETRIC COMPARISON: L9, L10, L11, L12")
    print("=" * 70)

    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    print("[2/3] Loading data...")
    text = load_wikitext("test", max_chars=MAX_CHARS)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    target_ids = input_ids[:, :WINDOW].cuda()
    print(f"  Input: {target_ids.shape}")

    print("\n[3/3] Extracting attention matrices...")

    # Extract softmax attention for each layer (original model, no swap)
    softmax_attns = {}
    with torch.no_grad():
        outputs = model(target_ids, output_attentions=True)
    for layer_idx in LAYERS:
        A = outputs.attentions[layer_idx]  # [B, H, N, N]
        softmax_attns[layer_idx] = A.float().cpu().numpy()
        print(f"  Softmax L{layer_idx}: shape={A.shape}")

    # Extract Focus Bubble attention for each layer
    focus_attns = {}
    for layer_idx in LAYERS:
        A = extract_attention(model, tokenizer, target_ids, layer_idx, EPSILON, TAU_ITERS)
        focus_attns[layer_idx] = A.float().cpu().numpy()
        print(f"  Focus  L{layer_idx}: shape={A.shape}")

    # Compute metrics
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # 1. Concentration ratios (Focus, per layer) — averaged across heads
    print("\n  --- Concentration Ratio (Focus, avg across heads) ---")
    print(f"  {'Layer':>8} {'CR_focus':>12} {'CR_softmax':>12} {'CR_diff':>12}")
    print("  " + "-" * 48)
    cr_focus = {}
    cr_softmax = {}
    for layer_idx in LAYERS:
        A_f_avg = focus_attns[layer_idx].mean(axis=(0, 1))  # [N, N]
        A_s_avg = softmax_attns[layer_idx].mean(axis=(0, 1))
        cr_f = concentration_ratio(A_f_avg, threshold=1e-3)
        cr_s = concentration_ratio(A_s_avg, threshold=1e-3)
        cr_focus[layer_idx] = cr_f
        cr_softmax[layer_idx] = cr_s
        print(f"  {'L'+str(layer_idx):>8} {cr_f:12.6f} {cr_s:12.6f} {cr_f - cr_s:+12.6f}")

    # 2. L2 diff Focus vs Softmax (per layer)
    print("\n  --- L2 diff: Focus vs Softmax (avg across heads) ---")
    print(f"  {'Layer':>8} {'L2_diff':>12} {'L2_base':>12} {'L2_ratio%':>10}")
    print("  " + "-" * 48)
    l2_ratios = {}
    for layer_idx in LAYERS:
        A_f_avg = focus_attns[layer_idx].mean(axis=(0, 1))
        A_s_avg = softmax_attns[layer_idx].mean(axis=(0, 1))
        l2_diff = np.linalg.norm(A_f_avg - A_s_avg)
        l2_base = np.linalg.norm(A_s_avg)
        l2_ratio = l2_diff / l2_base * 100
        l2_ratios[layer_idx] = l2_ratio
        print(f"  {'L'+str(layer_idx):>8} {l2_diff:12.6f} {l2_base:12.6f} {l2_ratio:9.2f}%")

    # 3. Pairwise L2 between Focus layers (L9 vs L10, L10 vs L11, L11 vs L12, L9 vs L12)
    print("\n  --- Pairwise L2 between Focus layers (avg across heads) ---")
    print(f"  {'Pair':>14} {'L2_diff':>12} {'L2_base_A':>12} {'L2_ratio%':>10}")
    print("  " + "-" * 54)
    for i in range(len(LAYERS)):
        for j in range(i + 1, len(LAYERS)):
            li = LAYERS[i]
            lj = LAYERS[j]
            A_i = focus_attns[li].mean(axis=(0, 1))
            A_j = focus_attns[lj].mean(axis=(0, 1))
            l2_diff = np.linalg.norm(A_i - A_j)
            l2_base = np.linalg.norm(A_i)
            l2_ratio = l2_diff / l2_base * 100
            pair_label = f"L{li} vs L{lj}"
            print(f"  {pair_label:>14} {l2_diff:12.6f} {l2_base:12.6f} {l2_ratio:9.2f}%")

    # 4. Pairwise L2 between Softmax layers (same pairs, for reference)
    print("\n  --- Pairwise L2 between Softmax layers (avg across heads, reference) ---")
    print(f"  {'Pair':>14} {'L2_diff':>12} {'L2_base_A':>12} {'L2_ratio%':>10}")
    print("  " + "-" * 54)
    for i in range(len(LAYERS)):
        for j in range(i + 1, len(LAYERS)):
            li = LAYERS[i]
            lj = LAYERS[j]
            A_i = softmax_attns[li].mean(axis=(0, 1))
            A_j = softmax_attns[lj].mean(axis=(0, 1))
            l2_diff = np.linalg.norm(A_i - A_j)
            l2_base = np.linalg.norm(A_i)
            l2_ratio = l2_diff / l2_base * 100
            pair_label = f"L{li} vs L{lj}"
            print(f"  {pair_label:>14} {l2_diff:12.6f} {l2_base:12.6f} {l2_ratio:9.2f}%")

    # 5. PPL for reference
    print("\n  --- PPL reference (from layer sweep, eps=0.001 tau=1) ---")
    print(f"  {'Layer':>8} {'PPL':>10} {'Delta%':>8} {'Gate':>6}")
    print("  " + "-" * 40)
    ppls = {9: (23.026, 2.28, "FAIL"), 10: (22.757, 1.08, "PASS"),
            11: (22.772, 1.15, "PASS"), 12: (22.706, 0.86, "PASS")}
    for layer_idx in LAYERS:
        ppl, delta, gate = ppls[layer_idx]
        print(f"  {'L'+str(layer_idx):>8} {ppl:10.3f} {delta:+7.2f}% {gate:>6}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
