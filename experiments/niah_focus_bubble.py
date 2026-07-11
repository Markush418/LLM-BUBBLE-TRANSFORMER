"""
NIAH (Needle-in-a-Haystack) Benchmark for Focus Bubble (Memory-Optimized)
==========================================================================
Tests retrieval accuracy at various context lengths (1K, 2K, 4K, 8K).
Memory-optimized for GTX 1650 (4.3GB VRAM).
"""

import os, sys, json, time, random
from pathlib import Path

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")

from qwen3_focus_bubble_wrapper import Qwen3FocusBubbleWrapper

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
SEED = 42

# NIAH parameters - reduced for GTX 1650
NEEDLE = "The secret password is 42."
QUESTION = "What is the secret password?"
CONTEXT_LENGTHS = [2048]
POSITIONS = ["middle"]
NUM_TRIALS = 1

LAYERS = [7, 10, 12]
EPSILON = 0.001
TAU_ITERS = 1
LAMBDA = 0.3


def build_context(target_len, needle, position, tokenizer):
    """Build haystack of target token length with needle at position.
    Uses a smaller, memory-efficient haystack generator."""
    
    # Simple repetitive text as haystack (memory efficient)
    base_text = "The quick brown fox jumps over the lazy dog. " * 20
    haystack_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    needle_tokens = tokenizer.encode(NEEDLE, add_special_tokens=False)
    
    if len(haystack_tokens) < target_len:
        repeat = (target_len // len(haystack_tokens)) + 1
        haystack_tokens = (haystack_tokens * repeat)[:target_len]
    
    max_context = target_len - len(NEEDLE.encode())
    context_tokens = haystack_tokens[:max_context]
    
    if position == "start":
        insert_idx = 0
    elif position == "middle":
        insert_idx = len(context_tokens) // 2
    else:  # end
        insert_idx = len(context_tokens)
    
    insert_idx = min(insert_idx, len(context_tokens))
    final_tokens = context_tokens[:insert_idx] + tokenizer.encode(NEEDLE) + context_tokens[insert_idx:]
    final_tokens = final_tokens[:target_len]
    
    return torch.tensor(final_tokens, dtype=torch.long).unsqueeze(0).cuda()


@torch.no_grad()
def evaluate_retrieval(model, tokenizer, input_ids, question, answer):
    """Check if model retrieves the needle correctly."""
    # Format as QA
    prompt = f"{tokenizer.decode(input_ids[0])}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    success = answer.lower() in generated.lower()
    return success, generated


def swap_layer(model, layer_idx, epsilon=0.001, tau_iters=1, lam=0.0, use_delta=False):
    """Swap a layer with Focus Bubble wrapper."""
    orig_attn = model.model.layers[layer_idx].self_attn
    wrapper = Qwen3FocusBubbleWrapper(
        original_attn=orig_attn,
        epsilon=epsilon,
        tau_iters=tau_iters,
        use_psi=True,
        use_delta=use_delta,
        lam=lam,
    ).cuda()
    model.model.layers[layer_idx].self_attn = wrapper
    return orig_attn


def restore_layer(model, layer_idx, orig_attn):
    model.model.layers[layer_idx].self_attn = orig_attn


def run_niah_for_config(model, tokenizer, config_name, layer_idx, use_delta, lam):
    """Run NIAH for a specific configuration."""
    print(f"\n  Testing {config_name} on L{layer_idx}...")
    
    orig_attn = swap_layer(model, layer_idx, use_delta=use_delta, lam=lam)
    
    results = {"config": config_name, "layer": layer_idx, "use_delta": use_delta, "lam": lam, "trials": []}
    
    for ctx_len in CONTEXT_LENGTHS:
        for pos in POSITIONS:
            for trial in range(NUM_TRIALS):
                try:
                    input_ids = build_context(ctx_len, NEEDLE, pos, tokenizer)
                    
                    success, generated = evaluate_retrieval(model, tokenizer, input_ids, QUESTION, "42")
                    
                    results["trials"].append({
                        "context_len": ctx_len,
                        "needle_pos": pos,
                        "trial": trial,
                        "success": success,
                        "generated": generated[:50],
                    })
                    
                    print(f"    ctx={ctx_len} pos={pos} trial={trial}: {'OK' if success else 'FAIL'}")
                    
                except Exception as e:
                    print(f"    ctx={ctx_len} pos={pos} trial={trial}: ERROR - {e}")
                    results["trials"].append({
                        "context_len": ctx_len,
                        "needle_pos": pos,
                        "trial": trial,
                        "success": False,
                        "error": str(e),
                    })
    
    restore_layer(model, layer_idx, orig_attn)
    return results


def main():
    out_dir = Path("results_real/niah_focus")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  NIAH BENCHMARK - FOCUS BUBBLE (Memory Optimized)")
    print("=" * 70)

    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    all_results = {}

    configs = [
        ("Softmax_Baseline", None, False, 0.0),
        ("Focus_L7", 7, False, 0.0),
        ("FocusDeltaNet_L7", 7, True, 0.3),
    ]

    for config_name, layer_idx, use_delta, lam in configs:
        if layer_idx is None:
            # Softmax baseline
            print(f"\n  Testing {config_name}...")
            results = {"config": config_name, "layer": None, "use_delta": False, "lam": 0.0, "trials": []}
            for ctx_len in CONTEXT_LENGTHS:
                for pos in POSITIONS:
                    for trial in range(NUM_TRIALS):
                        try:
                            input_ids = build_context(ctx_len, NEEDLE, pos, tokenizer)
                            success, generated = evaluate_retrieval(model, tokenizer, input_ids, QUESTION, "42")
                            results["trials"].append({
                                "context_len": ctx_len, "needle_pos": pos, "trial": trial,
                                "success": success, "generated": generated[:50],
                            })
                            print(f"    ctx={ctx_len} pos={pos} trial={trial}: {'OK' if success else 'FAIL'}")
                        except Exception as e:
                            print(f"    ctx={ctx_len} pos={pos} trial={trial}: ERROR - {e}")
                            results["trials"].append({
                                "context_len": ctx_len, "needle_pos": pos, "trial": trial,
                                "success": False, "error": str(e),
                            })
            all_results[config_name] = results
        else:
            results = run_niah_for_config(model, tokenizer, config_name, layer_idx, use_delta, lam)
            all_results[config_name] = results

    # Save results
    out_dir = Path("results_real/niah_focus")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"niah_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary
    print("\n" + "=" * 70)
    print("  NIAH SUMMARY")
    print("=" * 70)
    for config_name, results in all_results.items():
        if not results["trials"]:
            continue
        total = len(results["trials"])
        success = sum(1 for t in results["trials"] if t.get("success", False))
        rate = success / total * 100
        print(f"  {config_name:30s}: {success}/{total} = {rate:.1f}%")

    print("\n  Per-context breakdown:")
    for config_name, results in all_results.items():
        if not results["trials"]:
            continue
        print(f"\n  {config_name}:")
        for ctx_len in CONTEXT_LENGTHS:
            ctx_trials = [t for t in results["trials"] if t.get("context_len") == ctx_len]
            if ctx_trials:
                s = sum(1 for t in ctx_trials if t.get("success", False))
                print(f"    {ctx_len:5d} tokens: {s}/{len(ctx_trials)} = {s/len(ctx_trials)*100:.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    import torch
    import numpy as np
    from pathlib import Path
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()