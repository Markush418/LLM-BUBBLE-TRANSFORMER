"""
Per-head disaggregation: L9 vs L10
===================================
CR and L2_diff per individual head (16 heads), Focus vs Softmax.
No averaging across heads. L10 is control (passes gate).
"""

import os, sys
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import torch
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
LAYERS = [9, 10]


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
def extract_attention(model, target_ids, layer_idx, eps, tau):
    layer = model.model.layers[layer_idx]
    orig_attn = layer.self_attn
    wrapper = Qwen3FocusBubbleWrapper(
        original_attn=orig_attn, epsilon=eps, tau_iters=tau,
        use_psi=True, use_delta=False, lam=0.0,
    ).cuda()
    layer.self_attn = wrapper
    outputs = model(target_ids, output_attentions=True)
    A = outputs.attentions[layer_idx]  # [B, H, N, N]
    layer.self_attn = orig_attn
    return A


def main():
    print("=" * 70)
    print("  PER-HEAD DISAGGREGATION: L9 vs L10")
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
    target_ids = input_ids[:, :WINDOW].cuda()

    print("\n[3/3] Extracting attention matrices...")

    # Softmax for all layers (one forward pass)
    with torch.no_grad():
        outputs = model(target_ids, output_attentions=True)

    softmax_attns = {}
    for layer_idx in LAYERS:
        A = outputs.attentions[layer_idx]  # [B, H, N, N]
        softmax_attns[layer_idx] = A.float().cpu().numpy()
        print(f"  Softmax L{layer_idx}: {A.shape}")

    # Focus for each layer
    focus_attns = {}
    for layer_idx in LAYERS:
        A = extract_attention(model, target_ids, layer_idx, EPSILON, TAU_ITERS)
        focus_attns[layer_idx] = A.float().cpu().numpy()
        print(f"  Focus  L{layer_idx}: {A.shape}")

    # Per-head metrics
    for layer_idx in LAYERS:
        print("\n" + "=" * 70)
        print(f"  LAYER {layer_idx} - PER-HEAD BREAKDOWN")
        print("=" * 70)

        A_s = softmax_attns[layer_idx]  # [B=1, H=16, N, N]
        A_f = focus_attns[layer_idx]
        B, H, N, _ = A_s.shape

        print(f"\n  {'Head':>6} {'CR_soft':>10} {'CR_focus':>10} {'CR_diff':>10} {'L2_diff':>10} {'L2_base':>10} {'L2_ratio%':>10}")
        print("  " + "-" * 76)

        cr_s_list = []
        cr_f_list = []
        l2_ratio_list = []

        for h in range(H):
            # Per-head: [N, N]
            A_s_h = A_s[0, h]  # [N, N]
            A_f_h = A_f[0, h]  # [N, N]

            cr_s = concentration_ratio(A_s_h, threshold=1e-3)
            cr_f = concentration_ratio(A_f_h, threshold=1e-3)
            cr_diff = cr_f - cr_s

            l2_diff = np.linalg.norm(A_f_h - A_s_h)
            l2_base = np.linalg.norm(A_s_h)
            l2_ratio = l2_diff / l2_base * 100 if l2_base > 0 else 0.0

            cr_s_list.append(cr_s)
            cr_f_list.append(cr_f)
            l2_ratio_list.append(l2_ratio)

            print(f"  {h:>6d} {cr_s:10.6f} {cr_f:10.6f} {cr_diff:+10.6f} {l2_diff:10.6f} {l2_base:10.6f} {l2_ratio:9.2f}%")

        # Stats
        print(f"\n  {'mean':>6} {np.mean(cr_s_list):10.6f} {np.mean(cr_f_list):10.6f} {np.mean(cr_f_list)-np.mean(cr_s_list):+10.6f} {'':>10} {'':>10} {np.mean(l2_ratio_list):9.2f}%")
        print(f"  {'std':>6} {np.std(cr_s_list):10.6f} {np.std(cr_f_list):10.6f} {np.std(np.array(cr_f_list)-np.array(cr_s_list)):+10.6f} {'':>10} {'':>10} {np.std(l2_ratio_list):9.2f}%")
        print(f"  {'min':>6} {np.min(cr_s_list):10.6f} {np.min(cr_f_list):10.6f} {'':>10} {'':>10} {'':>10} {np.min(l2_ratio_list):9.2f}%")
        print(f"  {'max':>6} {np.max(cr_s_list):10.6f} {np.max(cr_f_list):10.6f} {'':>10} {'':>10} {'':>10} {np.max(l2_ratio_list):9.2f}%")

        # Find outliers: heads with L2_ratio > mean + 1.5*std
        mean_l2 = np.mean(l2_ratio_list)
        std_l2 = np.std(l2_ratio_list)
        threshold = mean_l2 + 1.5 * std_l2
        outliers = [(h, l2_ratio_list[h]) for h in range(H) if l2_ratio_list[h] > threshold]
        if outliers:
            print(f"\n  OUTLIER heads (L2_ratio > {threshold:.2f}%):")
            for h, ratio in outliers:
                print(f"    Head {h}: L2_ratio={ratio:.2f}%  CR_s={cr_s_list[h]:.6f}  CR_f={cr_f_list[h]:.6f}  CR_diff={cr_f_list[h]-cr_s_list[h]:+.6f}")
        else:
            print(f"\n  No outlier heads (threshold = {threshold:.2f}%)")

    # Direct comparison L9 vs L10 per head
    print("\n" + "=" * 70)
    print("  L9 vs L10 - PER-HEAD COMPARISON")
    print("=" * 70)
    print(f"\n  {'Head':>6} {'L9_L2%':>10} {'L10_L2%':>10} {'L9-L10':>10} {'L9_CRdiff':>10} {'L10_CRdiff':>10}")
    print("  " + "-" * 66)

    for h in range(16):
        A_s9 = softmax_attns[9][0, h]
        A_f9 = focus_attns[9][0, h]
        A_s10 = softmax_attns[10][0, h]
        A_f10 = focus_attns[10][0, h]

        l2_9 = np.linalg.norm(A_f9 - A_s9) / np.linalg.norm(A_s9) * 100
        l2_10 = np.linalg.norm(A_f10 - A_s10) / np.linalg.norm(A_s10) * 100
        cr_diff_9 = concentration_ratio(A_f9) - concentration_ratio(A_s9)
        cr_diff_10 = concentration_ratio(A_f10) - concentration_ratio(A_s10)

        print(f"  {h:>6d} {l2_9:9.2f}% {l2_10:9.2f}% {l2_9-l2_10:+9.2f} {cr_diff_9:+10.6f} {cr_diff_10:+10.6f}")

    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
