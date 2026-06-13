"""
Bubble Assignment Analysis per Layer
======================================
Passes real text through the model and records:
- Number of unique bubbles used per layer
- Bubble assignment entropy per layer
- Whether clustering structure varies across depth

Usage:
    python scripts/analyze_bubble_assignments.py
"""

import sys
import math
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

REPO_PATH = Path(r"C:\Users\negocio\Desktop\LLM-BUBBLE")
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
NUM_BUBBLES = 32
TOP_K = 64
EPS_STAR = 0.005
ROUTING_BONUS = 0.2

OUTPUT_FILE = Path("bubble_assignment_analysis.json")


def entropy(probs):
    p = probs[probs > 0]
    return -np.sum(p * np.log2(p))


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper

    # Load a small text sample
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())[:2000]

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[:, :512].to(DEVICE)
    n_tokens = input_ids.shape[1]

    print(f"Tokens: {n_tokens}")

    # Hook to capture assignments per layer
    layer_data = []

    for layer_idx in range(len(model.model.layers)):
        original_attn = model.model.layers[layer_idx].self_attn

        wrapper = Qwen3GQABubbleWrapper(
            original_attn=original_attn,
            num_bubbles=NUM_BUBBLES,
            top_k=TOP_K,
            eps=EPS_STAR,
            routing_bonus=ROUTING_BONUS,
            debug=False,
        )
        model.model.layers[layer_idx].self_attn = wrapper

    model.eval()

    # Register forward hooks to capture assignments
    assignment_cache = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if hasattr(module, '_last_assignments'):
                assignment_cache[layer_idx] = module._last_assignments.detach().cpu().numpy()
        return hook_fn

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Modify wrapper to store assignments
    for layer in model.model.layers:
        if isinstance(layer.self_attn, Qwen3GQABubbleWrapper):
            layer.self_attn._last_assignments = None

    # Monkey-patch forward to also save assignments
    original_forward = Qwen3GQABubbleWrapper.forward

    def patched_forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        # Re-run clustering to capture assignments
        B, N, D = args[0].shape if args else kwargs['hidden_states'].shape
        hidden_states = args[0] if args else kwargs['hidden_states']
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        K = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        if kwargs.get('position_embeddings') is not None:
            cos, sin = kwargs['position_embeddings']
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            _, K = apply_rotary_pos_emb(K, K, cos, sin)
            # We need Q for RoPE but K is what matters for clustering
            # Actually RoPE was applied to both Q and K in the original forward
            # For clustering we use K after RoPE
        K_flat = K.float().mean(dim=1)
        centroids_dev = self.centroids.to(K_flat.device).unsqueeze(0).expand(B, -1, -1)
        dists = torch.cdist(K_flat, centroids_dev)
        assignments = dists.argmin(dim=-1)
        self._last_assignments = assignments
        return result

    Qwen3GQABubbleWrapper.forward = patched_forward

    # Run forward pass
    with torch.no_grad():
        _ = model(input_ids)

    # Analyze assignments
    print(f"\n{'='*60}")
    print(f" Bubble Assignment Analysis (bonus={ROUTING_BONUS}, bubbles={NUM_BUBBLES})")
    print(f"{'='*60}")
    print(f" {'Layer':>6} {'Unique':>8} {'Entropy':>10} {'Max%':>8} {'Top3%':>8}")
    print(f"{'-'*60}")

    all_results = []
    for layer_idx in range(len(model.model.layers)):
        if layer_idx not in assignment_cache:
            # Fallback: recompute from wrapper's stored assignments
            wrapper = model.model.layers[layer_idx].self_attn
            if hasattr(wrapper, '_last_assignments') and wrapper._last_assignments is not None:
                assigns = wrapper._last_assignments
            else:
                continue
        else:
            assigns = assignment_cache[layer_idx]

        if isinstance(assigns, torch.Tensor):
            assigns = assigns.detach().cpu().numpy()

        assigns = assigns.flatten()  # [B*N]
        unique = len(np.unique(assigns))
        counts = np.bincount(assigns, minlength=NUM_BUBBLES)
        probs = counts / counts.sum()
        ent = entropy(probs)
        max_pct = probs.max() * 100
        top3_pct = np.sort(probs)[-3:][::-1].sum() * 100

        print(f" {layer_idx:>6} {unique:>8} {ent:>10.3f} {max_pct:>7.2f}% {top3_pct:>7.2f}%")
        all_results.append({
            "layer": layer_idx,
            "unique_bubbles": int(unique),
            "entropy": float(ent),
            "max_bubble_pct": float(max_pct),
            "top3_pct": float(top3_pct),
            "max_entropy": float(math.log2(NUM_BUBBLES)),
            "entropy_ratio": float(ent / math.log2(NUM_BUBBLES)),
        })

    # Summary
    avg_unique = np.mean([r["unique_bubbles"] for r in all_results])
    avg_entropy_ratio = np.mean([r["entropy_ratio"] for r in all_results])
    print(f"{'-'*60}")
    print(f" Avg unique bubbles: {avg_unique:.1f} / {NUM_BUBBLES}")
    print(f" Avg entropy ratio:  {avg_entropy_ratio:.3f} (1.0 = uniform, 0.0 = single cluster)")
    print(f" Max possible entropy: {math.log2(NUM_BUBBLES):.3f} bits")
    print(f"{'='*60}\n")

    summary = {
        "config": {
            "model": MODEL_ID,
            "num_bubbles": NUM_BUBBLES,
            "routing_bonus": ROUTING_BONUS,
            "eps": EPS_STAR,
            "n_tokens": int(n_tokens),
        },
        "summary": {
            "avg_unique_bubbles": float(avg_unique),
            "avg_entropy_ratio": float(avg_entropy_ratio),
            "max_entropy_bits": float(math.log2(NUM_BUBBLES)),
        },
        "per_layer": all_results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"-> Saved to {OUTPUT_FILE}")

    # Cleanup
    for h in hooks:
        h.remove()
    Qwen3GQABubbleWrapper.forward = original_forward


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
