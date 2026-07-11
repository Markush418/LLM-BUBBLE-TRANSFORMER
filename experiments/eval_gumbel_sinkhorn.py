"""Evaluate Gumbel-Sinkhorn attention (tau=0.1) on Qwen3-0.6B-Base with WikiText-2.

Compares:
  1. Baseline softmax
  2. Per-layer Gumbel-Sinkhorn replacement (layers [3,7,11,15,19,23])
  3. Hybrid mode: DeltaNet base + Gumbel-Sinkhorn post-processing

Outputs JSON to stdout.
"""

import os
import sys
import gc
import json
import math
import time

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")

from attention_variants.gumbel_sinkhorn import gumbel_sinkhorn_attention
from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
TARGET_LAYERS = [3, 7, 11, 15, 19, 23]
WINDOW = 64
STRIDE = 64
MAX_CHARS = 15_000
SEED = 42


# ---------------------------------------------------------------------------
# Gumbel-Sinkhorn wrapper for Qwen3Attention (drop-in replacement)
# ---------------------------------------------------------------------------
class GumbelSinkhornAttentionWrapper(nn.Module):
    """Drop-in replacement for Qwen3Attention using Gumbel-Sinkhorn."""

    def __init__(self, original_attn, epsilon=0.1, tau=0.1, n_sinkhorn_iters=5):
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm

        self.num_heads = original_attn.config.num_attention_heads
        self.num_kv_heads = original_attn.config.num_key_value_heads
        self.head_dim = original_attn.config.head_dim
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = original_attn.scaling

        self.epsilon = epsilon
        self.tau = tau
        self.n_sinkhorn_iters = n_sinkhorn_iters

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        B, N, D = hidden_states.shape
        device = hidden_states.device
        orig_dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # GQA expansion
        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)
            V = V.repeat_interleave(self.kv_groups, dim=1)

        # RoPE
        if position_embeddings is not None:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos.to(Q.dtype), sin.to(K.dtype))

        # Gumbel-Sinkhorn attention (compute in float32 for stability)
        out = gumbel_sinkhorn_attention(
            Q.float(), K.float(), V.float(),
            epsilon=self.epsilon,
            tau=self.tau,
            n_sinkhorn_iters=self.n_sinkhorn_iters,
            causal=True,
        )
        out = out.to(orig_dtype)

        out = out.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        if output_attentions:
            return out, None
        return out, None


# ---------------------------------------------------------------------------
# Hybrid wrapper: DeltaNet base + Gumbel-Sinkhorn post-processing
# ---------------------------------------------------------------------------
class HybridDeltaNetGumbelSinkhornWrapper(nn.Module):
    """DeltaNet base + Gumbel-Sinkhorn post-processing (siri_mode=soft blend)."""

    def __init__(self, original_attn, epsilon=0.1, tau=0.1, n_sinkhorn_iters=5,
                 siri_alpha=0.7):
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm

        self.num_heads = original_attn.config.num_attention_heads
        self.num_kv_heads = original_attn.config.num_key_value_heads
        self.head_dim = original_attn.config.head_dim
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.hidden_size = original_attn.config.hidden_size
        self.scaling = original_attn.scaling
        self.out_head_dim = self.hidden_size // self.num_heads

        self.epsilon = epsilon
        self.tau = tau
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.siri_alpha = siri_alpha

    def _delta_rule_step(self, Q_n, K_n, V_n):
        """Recurrent DeltaNet pass (linear O(N) attention)."""
        B, H, N, D_h = Q_n.shape
        out_delta = torch.zeros(B, H, N, D_h, dtype=Q_n.dtype, device=Q_n.device)
        S = torch.zeros(B, H, D_h, D_h, dtype=Q_n.dtype, device=Q_n.device)
        norm_decay = 1.0 - 1.0 / max(N, 2)
        for t in range(N):
            k_t = K_n[:, :, t]
            v_t = V_n[:, :, t]
            q_t = Q_n[:, :, t]
            v_old = torch.einsum("bhij,bhj->bhi", S, k_t)
            delta = v_t - v_old
            S = norm_decay * S + torch.einsum("bhj,bhi->bhij", k_t, delta)
            out_delta[:, :, t] = torch.einsum("bhij,bhj->bhi", S, q_t)
        return out_delta

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        B, N, D = hidden_states.shape
        device = hidden_states.device
        orig_dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)
            V = V.repeat_interleave(self.kv_groups, dim=1)

        # RoPE
        if position_embeddings is not None:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos.to(Q.dtype), sin.to(K.dtype))

        Q_f = Q.float()
        K_f = K.float()
        V_f = V.float()

        # Normalize for DeltaNet
        Q_n = Q_f / Q_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        K_n = K_f / K_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        V_n = V_f / V_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # DeltaNet base
        out_delta = self._delta_rule_step(Q_n, K_n, V_n)

        # Softmax attention output (standard baseline)
        causal_mask = torch.triu(torch.full((N, N), float("-inf"), device=device), diagonal=1)
        attn_scores = (Q_f @ K_f.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        A_softmax = torch.softmax(attn_scores, dim=-1)
        out_softmax = (A_softmax @ V_f)  # [B, H, N, D_h]

        # Gumbel-Sinkhorn output
        out_gs = gumbel_sinkhorn_attention(
            Q_f, K_f, V_f,
            epsilon=self.epsilon,
            tau=self.tau,
            n_sinkhorn_iters=self.n_sinkhorn_iters,
            causal=True,
        )

        # Blend outputs: (1-alpha) * softmax + alpha * GS
        out_siri = ((1.0 - self.siri_alpha) * out_softmax + self.siri_alpha * out_gs)

        # Interpolate DeltaNet and SIRI (50/50)
        out = 0.5 * out_delta + 0.5 * out_siri
        out = out.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim).to(orig_dtype)

        out = self.o_proj(out)

        if output_attentions:
            return out, A_blend
        return out, None


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_wikitext_test_text(max_chars=None):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars is not None:
        text = text[:max_chars]
    return text


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_perplexity(model, input_ids, window=WINDOW, stride=STRIDE, device="cuda"):
    n_tokens = input_ids.shape[1]
    nlls = []
    n_tokens_counted = 0
    for begin_loc in range(0, n_tokens - window, stride):
        end_loc = begin_loc + window
        target_ids = input_ids[:, begin_loc:end_loc].to(device)
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


# ---------------------------------------------------------------------------
# Swap / restore helpers
# ---------------------------------------------------------------------------
def swap_layer_gumbel_sinkhorn(model, layer_idx, epsilon, tau, n_sinkhorn_iters):
    layer = model.model.layers[layer_idx]
    orig = layer.self_attn
    wrapper = GumbelSinkhornAttentionWrapper(
        orig, epsilon=epsilon, tau=tau, n_sinkhorn_iters=n_sinkhorn_iters,
    ).cuda()
    layer.self_attn = wrapper
    return orig


def swap_layer_hybrid(model, layer_idx, epsilon, tau, n_sinkhorn_iters, siri_alpha):
    layer = model.model.layers[layer_idx]
    orig = layer.self_attn
    wrapper = HybridDeltaNetGumbelSinkhornWrapper(
        orig, epsilon=epsilon, tau=tau, n_sinkhorn_iters=n_sinkhorn_iters,
        siri_alpha=siri_alpha,
    ).cuda()
    layer.self_attn = wrapper
    return orig


def restore_layer(model, layer_idx, original_attn):
    model.model.layers[layer_idx].self_attn = original_attn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results = {}

    # Load model
    print("Loading Qwen3-0.6B-Base...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.cuda.empty_cache()
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", file=sys.stderr)

    # Load data
    print(f"Loading WikiText-2 test (max {MAX_CHARS} chars)...", file=sys.stderr)
    text = load_wikitext_test_text(max_chars=MAX_CHARS)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    n_tokens = input_ids.shape[1]
    n_windows = (n_tokens - WINDOW) // STRIDE
    print(f"Tokens: {n_tokens}, Windows: {n_windows}", file=sys.stderr)

    # --- Baseline ---
    print("Evaluating baseline softmax...", file=sys.stderr)
    torch.cuda.empty_cache()
    t0 = time.time()
    ppl_baseline = eval_perplexity(model, input_ids, WINDOW, STRIDE)
    dt_baseline = time.time() - t0
    torch.cuda.empty_cache()
    print(f"  Baseline PPL = {ppl_baseline:.3f}  ({dt_baseline:.1f}s)", file=sys.stderr)
    results["baseline"] = {
        "ppl": ppl_baseline,
        "time_s": round(dt_baseline, 2),
    }

    # --- Per-layer Gumbel-Sinkhorn ---
    print("\nPer-layer Gumbel-Sinkhorn evaluation...", file=sys.stderr)
    for layer_idx in TARGET_LAYERS:
        key = f"gs_layer_{layer_idx}"
        try:
            torch.cuda.empty_cache()
            orig = swap_layer_gumbel_sinkhorn(model, layer_idx, epsilon=0.1, tau=0.1, n_sinkhorn_iters=5)
            t0 = time.time()
            ppl = eval_perplexity(model, input_ids, WINDOW, STRIDE)
            dt = time.time() - t0
            restore_layer(model, layer_idx, orig)
            del orig
            gc.collect()
            torch.cuda.empty_cache()
            delta = ppl - ppl_baseline
            print(f"  L{layer_idx:02d}: PPL={ppl:.3f}  dPPL={delta:+.3f}  ({dt:.1f}s)", file=sys.stderr)
            results[key] = {
                "ppl": ppl,
                "delta_ppl": round(delta, 3),
                "epsilon": 0.1,
                "tau": 0.1,
                "n_sinkhorn_iters": 5,
                "time_s": round(dt, 2),
            }
        except Exception as e:
            print(f"  L{layer_idx:02d}: ERROR — {e}", file=sys.stderr)
            results[key] = {"ppl": float("nan"), "error": str(e)}
            try:
                restore_layer(model, layer_idx, orig)
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    # --- Hybrid: DeltaNet + Gumbel-Sinkhorn ---
    print("\nHybrid mode (DeltaNet + Gumbel-Sinkhorn, alpha=0.7)...", file=sys.stderr)
    for layer_idx in TARGET_LAYERS:
        key = f"hybrid_layer_{layer_idx}"
        try:
            torch.cuda.empty_cache()
            orig = swap_layer_hybrid(model, layer_idx, epsilon=0.1, tau=0.1,
                                     n_sinkhorn_iters=5, siri_alpha=0.7)
            t0 = time.time()
            ppl = eval_perplexity(model, input_ids, WINDOW, STRIDE)
            dt = time.time() - t0
            restore_layer(model, layer_idx, orig)
            del orig
            gc.collect()
            torch.cuda.empty_cache()
            delta = ppl - ppl_baseline
            print(f"  L{layer_idx:02d}: PPL={ppl:.3f}  dPPL={delta:+.3f}  ({dt:.1f}s)", file=sys.stderr)
            results[key] = {
                "ppl": ppl,
                "delta_ppl": round(delta, 3),
                "mode": "hybrid_deltanet_gumbel_sinkhorn",
                "epsilon": 0.1,
                "tau": 0.1,
                "siri_alpha": 0.7,
                "time_s": round(dt, 2),
            }
        except Exception as e:
            print(f"  L{layer_idx:02d}: ERROR — {e}", file=sys.stderr)
            results[key] = {"ppl": float("nan"), "error": str(e)}
            try:
                restore_layer(model, layer_idx, orig)
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    # --- Summary ---
    results["summary"] = {
        "model": MODEL_ID,
        "dataset": "wikitext-2-raw-v1",
        "max_chars": MAX_CHARS,
        "window": WINDOW,
        "stride": STRIDE,
        "n_tokens": n_tokens,
        "n_windows": n_windows,
        "target_layers": TARGET_LAYERS,
        "baseline_ppl": ppl_baseline,
    }

    # Output JSON to stdout
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
