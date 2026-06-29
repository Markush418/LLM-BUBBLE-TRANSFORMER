"""Perplexity benchmark: Hybrid Attention vs Baseline vs Plateau on Qwen3-0.6B.

Compares language modeling perplexity on WikiText-2 with three configurations:
  1. Baseline    - standard softmax attention (Qwen3 native)
  2. Plateau     - SIRI post-processing only (no DeltaNet)
  3. Hybrid      - DeltaNet + SIRI + Power Diagram psi (lambda sweep)

PPL formula: exp( -1/N * sum log P(x_i | x_<i) )  evaluated per window
"""

import os
import sys
import json
import math
import argparse
import time
from pathlib import Path

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")

from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_WINDOW = 256         # tokens per evaluation window
DEFAULT_STRIDE = 256         # non-overlapping windows
DEFAULT_LAYERS = None        # None = ALL 28 layers (full swap)
SEED = 42

# Hybrid hyperparameters
LAMBDA_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
EPSILON = 0.01


# ----------------------------------------------------------------------
# Data loader
# ----------------------------------------------------------------------
def load_wikitext_test_text(max_chars=None):
    """Load WikiText-2 test split as a single concatenated string."""
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars is not None:
        text = text[:max_chars]
    return text


# ----------------------------------------------------------------------
# PPL evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_perplexity(model, input_ids, window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE, device="cuda"):
    """Sliding-window perplexity evaluation (standard LM eval).

    Splits tokenized text into non-overlapping windows of size `window`
    with `stride` stride, computes cross-entropy loss per window, returns
    exp(mean(NLL)).
    """
    n_tokens = input_ids.shape[1]
    nlls = []
    n_tokens_counted = 0

    for begin_loc in range(0, n_tokens - window, stride):
        end_loc = begin_loc + window
        target_ids = input_ids[:, begin_loc:end_loc].to(device)

        # Forward; model returns logits.
        outputs = model(target_ids)
        logits = outputs.logits  # (1, window, vocab)

        # Standard LM shift: predict token i+1 from token i.
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


# ----------------------------------------------------------------------
# Swap attention layers in-place
# ----------------------------------------------------------------------
def swap_attention_layers(model, lam, epsilon, layer_indices=None, siri_mode="classical",
                          siri_beta=5.0, siri_alpha=0.3):
    """Swap specified layers' self_attn to HybridWrapper.

    Returns a list of (layer_idx, original_self_attn) for restoration.

    NOTE: The wrapper keeps its internal SIRI/PD components in float32 for
    numerical stability, while the wrapped projections stay in the model's
    dtype (float16). Do NOT cast the whole wrapper to fp16 — that downcasts
    _pd.W_psi and breaks the matmul.

    Args:
        siri_mode: "classical" | "chiller" | "sparse" | "soft"
            - classical: pure Sinkhorn (-C/eps)
            - chiller: Sinkhorn on QK^T/sqrt(d) scores scaled by siri_beta (peaked!)
            - sparse: ReLU + Sinkhorn (very sparse, doubly-stoch)
            - soft: blend Sinkhorn with softmax(siri_alpha blend weight)
    """
    if layer_indices is None:
        layer_indices = range(len(model.model.layers))

    original_self_attns = []
    for i in layer_indices:
        layer = model.model.layers[i]
        orig_attn = layer.self_attn
        wrapper = Qwen3HybridGQABubbleWrapper(
            original_attn=orig_attn,
            epsilon=epsilon,
            lam=lam,
            siri_mode=siri_mode,
            siri_beta=siri_beta,
            siri_alpha=siri_alpha,
        ).cuda()
        layer.self_attn = wrapper
        original_self_attns.append((i, orig_attn))
    return original_self_attns


def restore_attention_layers(model, original_self_attns):
    for i, orig_attn in original_self_attns:
        model.model.layers[i].self_attn = orig_attn


# ----------------------------------------------------------------------
# Main benchmark
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=200_000,
                        help="Max chars from WikiText-2 to evaluate on")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to swap (default: all 28)")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[EPSILON])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--siri-modes", type=str, nargs="+", default=["classical"],
                        help="SIRI variant: classical/chiller/sparse/soft")
    parser.add_argument("--siri-beta", type=float, default=5.0)
    parser.add_argument("--siri-alpha", type=float, default=0.3)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-plateau", action="store_true",
                        help="Skip pure SIRI (lambda=0) and pure DeltaNet (lambda=1)")
    parser.add_argument("--output-dir", default="results_real/perplexity")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}")
    print(f"Layers: {args.layers or 'ALL 28'}")
    print(f"Lambdas: {args.lambdas}")
    print(f"Epsilons: {args.epsilons}")
    print(f"Window: {args.window}, Stride: {args.stride}, Max chars: {args.max_chars}")
    print("=" * 60)

    # ----- Load model -----
    print("\n[1/4] Loading Qwen3-0.6B-Base...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",  # required for our custom attention
    )
    model.eval()
    print(f"      VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ----- Load data -----
    print(f"\n[2/4] Loading WikiText-2 test (max {args.max_chars} chars)...")
    text = load_wikitext_test_text(max_chars=args.max_chars)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    n_tokens = input_ids.shape[1]
    n_windows = (n_tokens - args.window) // args.stride
    print(f"      {n_tokens} tokens, {n_windows} evaluation windows")

    results = {}

    # ----- Baseline -----
    if not args.skip_baseline:
        print("\n[3/4] Baseline (standard softmax)...")
        t0 = time.time()
        ppl_baseline = eval_perplexity(model, input_ids, args.window, args.stride)
        dt = time.time() - t0
        print(f"      PPL = {ppl_baseline:.3f}  ({dt:.1f}s)")
        results["baseline"] = {"ppl": ppl_baseline, "time_s": dt}

    # ----- Lambda sweep -----
    print(f"\n[4/4] Hybrid sweep ({len(args.lambdas)} lambdas × {len(args.siri_modes)} modes)...")
    for siri_mode in args.siri_modes:
        for lam in args.lambdas:
            if args.skip_plateau and lam in [0.0, 1.0]:
                print(f"  lambda={lam} skipped")
                continue
            for eps in args.epsilons:
                key = f"hybrid_lam{lam}_eps{eps}_{siri_mode}"
                print(f"  -> lambda={lam}, epsilon={eps}, mode={siri_mode}...")
                original = swap_attention_layers(
                    model, lam=lam, epsilon=eps,
                    layer_indices=args.layers,
                    siri_mode=siri_mode,
                    siri_beta=args.siri_beta,
                    siri_alpha=args.siri_alpha,
                )
                t0 = time.time()
                try:
                    ppl = eval_perplexity(model, input_ids, args.window, args.stride)
                except Exception as e:
                    print(f"      ERROR: {e}")
                    ppl = float('nan')
                dt = time.time() - t0
                restore_attention_layers(model, original)
                print(f"      PPL = {ppl:.3f}  ({dt:.1f}s)")
                results[key] = {
                    "ppl": ppl, "lambda": lam, "epsilon": eps,
                    "siri_mode": siri_mode, "time_s": dt,
                }

    # ----- Save results -----
    out_file = out_dir / "ppl_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # ----- Summary table -----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'config':<30} {'PPL':>10} {'time(s)':>10}")
    print("-" * 60)
    for key, r in results.items():
        ppl_str = f"{r['ppl']:.3f}" if r['ppl'] == r['ppl'] else "NaN"
        print(f"{key:<30} {ppl_str:>10} {r['time_s']:>10.1f}")
    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()