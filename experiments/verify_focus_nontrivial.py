"""
Verify Focus Bubble Non-Triviality
===================================
Compara attention matrices:
  - Softmax original (L12)
  - FocusBubble L12 eps=0.001 tau=1

Calcula:
  (a) Diferencia L2 entre ambas matrices
  (b) Concentration ratio de cada una (fraccion de entradas > 1e-3)
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


def load_wikitext(split, max_chars=None):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars:
        text = text[:max_chars]
    return text


def concentration_ratio(A_np, threshold=1e-3):
    """Fraccion de entradas > threshold."""
    total = A_np.size
    above = (A_np > threshold).sum()
    return above / total


def main():
    print("=" * 70)
    print("  VERIFY FOCUS BUBBLE NON-TRIVIALITY")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    # Load data
    print("[2/4] Loading data...")
    text = load_wikitext("test", max_chars=MAX_CHARS)
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    # Take first window of 256 tokens (same as sweep)
    target_ids = input_ids[:, :WINDOW].cuda()
    print(f"  Input: {target_ids.shape}")

    # Step 1: Get softmax attention from original model
    print("\n[3/4] Extracting softmax attention (L12)...")
    with torch.no_grad():
        outputs = model(target_ids, output_attentions=True)
    # outputs.attentions is a tuple of [B, H, N, N] per layer
    # Layer 12 is index 12
    A_softmax = outputs.attentions[12]  # [B, H, N, N]
    print(f"  A_softmax shape: {A_softmax.shape}")
    print(f"  A_softmax dtype: {A_softmax.dtype}")

    # Average across heads and batch for a single [N, N] matrix
    A_softmax_avg = A_softmax.mean(dim=(0, 1)).float().cpu().numpy()  # [N, N]
    print(f"  A_softmax_avg shape: {A_softmax_avg.shape}")

    # Also keep per-head for comparison
    A_softmax_head0 = A_softmax[0, 0].float().cpu().numpy()  # [N, N] head 0

    # Step 2: Get Focus Bubble attention
    print("\n[4/4] Extracting Focus Bubble attention (L12, eps=0.001, tau=1)...")
    layer_idx = 12
    layer = model.model.layers[layer_idx]
    orig_attn = layer.self_attn

    wrapper = Qwen3FocusBubbleWrapper(
        original_attn=orig_attn,
        epsilon=0.001,
        tau_iters=1,
        use_psi=True,
        use_delta=False,
        lam=0.0,
    ).cuda()
    layer.self_attn = wrapper

    with torch.no_grad():
        outputs_focus = model(target_ids, output_attentions=True)

    # The wrapper returns attn_weights in the second position
    # But output_attentions=True should capture them
    # Let's check what we get
    if outputs_focus.attentions is not None:
        A_focus = outputs_focus.attentions[12]  # [B, H, N, N]
        print(f"  A_focus from output_attentions: {A_focus.shape}")
    else:
        # Need to extract manually - run forward with output_attentions
        # The wrapper returns (out, attn_weights) when output_attentions=True
        # But model.forward may not pass it through correctly
        # Let's call the layer directly
        print("  output_attentions not captured, calling layer directly...")
        hidden_states = model.model.embed_tokens(target_ids)
        # Get position embeddings
        position_ids = torch.arange(WINDOW, device=target_ids.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        # Build causal mask
        causal_mask = torch.triu(
            torch.full((WINDOW, WINDOW), float("-inf"), device=hidden_states.device, dtype=torch.float32),
            diagonal=1,
        )

        # Run layers 0-11 normally
        for i in range(layer_idx):
            hidden_states = model.model.layers[i](
                hidden_states,
                attention_mask=causal_mask.unsqueeze(0).unsqueeze(0),
                position_embeddings=position_embeddings,
            )[0]

        # Run layer 12 with output_attentions
        out_layer = model.model.layers[layer_idx](
            hidden_states,
            attention_mask=causal_mask.unsqueeze(0).unsqueeze(0),
            position_embeddings=position_embeddings,
            output_attentions=True,
        )
        # out_layer[0] = hidden_states, out_layer[1] = attn_weights
        A_focus = out_layer[1]  # [B, H, N, N] or [B, H, N, N]
        print(f"  A_focus from layer call: {A_focus.shape if A_focus is not None else 'None'}")

    # Restore original attention
    layer.self_attn = orig_attn

    if A_focus is None:
        print("ERROR: Could not extract Focus Bubble attention weights")
        return

    # Average across heads and batch
    A_focus_avg = A_focus.mean(dim=(0, 1)).float().cpu().numpy()  # [N, N]
    A_focus_head0 = A_focus[0, 0].float().cpu().numpy()  # [N, N] head 0

    # Step 3: Compute metrics
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # (a) L2 difference (averaged matrix)
    l2_diff_avg = np.linalg.norm(A_softmax_avg - A_focus_avg)
    l2_baseline_avg = np.linalg.norm(A_softmax_avg)
    l2_ratio_avg = l2_diff_avg / l2_baseline_avg * 100

    # Also for head 0
    l2_diff_head0 = np.linalg.norm(A_softmax_head0 - A_focus_head0)
    l2_baseline_head0 = np.linalg.norm(A_softmax_head0)
    l2_ratio_head0 = l2_diff_head0 / l2_baseline_head0 * 100

    # (b) Concentration ratios
    cr_softmax_avg = concentration_ratio(A_softmax_avg, threshold=1e-3)
    cr_focus_avg = concentration_ratio(A_focus_avg, threshold=1e-3)
    cr_softmax_head0 = concentration_ratio(A_softmax_head0, threshold=1e-3)
    cr_focus_head0 = concentration_ratio(A_focus_head0, threshold=1e-3)

    print(f"\n  --- Averaged across heads (mean) ---")
    print(f"  L2_diff:           {l2_diff_avg:.6f}")
    print(f"  L2_baseline:       {l2_baseline_avg:.6f}")
    print(f"  L2_ratio:          {l2_ratio_avg:.2f}%")
    print(f"  CR_softmax:        {cr_softmax_avg:.6f}")
    print(f"  CR_focus:          {cr_focus_avg:.6f}")

    print(f"\n  --- Head 0 only ---")
    print(f"  L2_diff:           {l2_diff_head0:.6f}")
    print(f"  L2_baseline:       {l2_baseline_head0:.6f}")
    print(f"  L2_ratio:          {l2_ratio_head0:.2f}%")
    print(f"  CR_softmax:        {cr_softmax_head0:.6f}")
    print(f"  CR_focus:          {cr_focus_head0:.6f}")

    print(f"\n  --- Summary numbers ---")
    print(f"  L2_diff (avg):     {l2_diff_avg:.6f}")
    print(f"  L2_ratio (avg):    {l2_ratio_avg:.2f}%")
    print(f"  CR_softmax (avg):  {cr_softmax_avg:.6f}")
    print(f"  CR_focus (avg):    {cr_focus_avg:.6f}")

    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
