"""
BT V5 Perplexity: Row-Stochastic Only (no Sinkhorn column normalization)
========================================================================
Tests if the PPL degradation is caused by doubly-stochastic Sinkhorn
or by the SIRI geometric cost mechanism itself.

If row-stochastic-only passes the gate, the issue is Sinkhorn.
If it still fails, the issue is fundamental to the geometric cost.
"""

import os
import sys
import json
import math
import time
from pathlib import Path

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
SEED = 42
MAX_CHARS = 50_000
WINDOW = 256
STRIDE = 256
SAFE_LAYERS = [3, 7, 11, 15, 19, 23]


def load_wikitext_test_text(max_chars=None):
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    lines = [r['text'] for r in ds if r['text'].strip()]
    text = '\n\n'.join(lines)
    if max_chars is not None:
        text = text[:max_chars]
    return text


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


class RowStochasticSIRI(nn.Module):
    """SIRI with row-stochastic only (no Sinkhorn column normalization).

    This is a causal-compatible variant: geometric cost + row normalization.
    """

    def __init__(self, original_attn, epsilon=0.01):
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
        self.hidden_size = original_attn.config.hidden_size
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.epsilon = epsilon

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

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        orig_dtype = hidden_states.dtype
        Q = Q.float()
        K = K.float()
        V = V.float()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            try:
                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
                Q, K = apply_rotary_pos_emb(Q, K, cos.to(Q.dtype), sin.to(K.dtype))
            except ImportError:
                cos_u = cos.unsqueeze(1)
                sin_u = sin.unsqueeze(1)
                d = Q.shape[-1]
                Q1, Q2 = Q[..., :d//2], Q[..., d//2:]
                K1, K2 = K[..., :d//2], K[..., d//2:]
                Q = torch.cat([Q1 * cos_u - Q2 * sin_u, Q2 * cos_u + Q1 * sin_u], dim=-1)
                K = torch.cat([K1 * cos_u - K2 * sin_u, K2 * cos_u + K1 * sin_u], dim=-1)

        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)
            V = V.repeat_interleave(self.kv_groups, dim=1)

        Q_norm = Q / Q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        K_norm = K / K.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        Q_sq = (Q_norm * Q_norm).sum(dim=-1, keepdim=True)
        K_sq = (K_norm * K_norm).sum(dim=-1, keepdim=True)
        C = (Q_sq + K_sq.transpose(-2, -1) - 2.0 * Q_norm @ K_norm.transpose(-2, -1)).clamp(min=0.0)

        C_min = C.amin(dim=(-2, -1), keepdim=True)
        C_max = C.amax(dim=(-2, -1), keepdim=True)
        C = (C - C_min) / (C_max - C_min + 1e-10)

        log_S = -C / self.epsilon

        # Apply causal mask BEFORE row-stochastic normalization
        if attention_mask is not None:
            causal_2d = attention_mask if attention_mask.dim() == 2 else attention_mask[0, 0]
            if causal_2d.shape[-1] != N:
                causal_2d = causal_2d[..., -N:]
            log_S = log_S + causal_2d.unsqueeze(0).unsqueeze(0)
        else:
            causal_mask = torch.triu(torch.full((N, N), float("-inf"), device=Q.device, dtype=Q.dtype), diagonal=1)
            log_S = log_S + causal_mask.unsqueeze(0).unsqueeze(0)

        log_S = log_S.clamp(min=-500.0, max=50.0)

        # Row-stochastic ONLY (no column normalization)
        A = log_S.exp()
        A = torch.nan_to_num(A, nan=0.0, posinf=1e10, neginf=0.0)
        row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        A = A / row_sums

        out_siri = (A @ V).transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)
        out_siri = out_siri.to(orig_dtype)
        out = self.o_proj(out_siri)

        if output_attentions:
            return out, A
        return out, None


class SoftmaxBaseline(nn.Module):
    """Standard softmax attention (baseline)."""

    def __init__(self, original_attn):
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
        self.hidden_size = original_attn.config.hidden_size
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = original_attn.scaling

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        **kwargs,
    ):
        B, N, D = hidden_states.shape
        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            try:
                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
                Q, K = apply_rotary_pos_emb(Q, K, cos.to(Q.dtype), sin.to(K.dtype))
            except ImportError:
                cos_u = cos.unsqueeze(1)
                sin_u = sin.unsqueeze(1)
                d = Q.shape[-1]
                Q1, Q2 = Q[..., :d//2], Q[..., d//2:]
                K1, K2 = K[..., :d//2], K[..., d//2:]
                Q = torch.cat([Q1 * cos_u - Q2 * sin_u, Q2 * cos_u + Q1 * sin_u], dim=-1)
                K = torch.cat([K1 * cos_u - K2 * sin_u, K2 * cos_u + K1 * sin_u], dim=-1)

        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)
            V = V.repeat_interleave(self.kv_groups, dim=1)

        attn_weights = (Q @ K.transpose(-2, -1)) * self.scaling
        causal_mask = torch.triu(torch.full((N, N), float("-inf"), device=Q.device, dtype=Q.dtype), diagonal=1)
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)
        out = self.o_proj(out.to(hidden_states.dtype))
        return out, None


def swap_layers(model, wrapper_class, layer_indices, epsilon=0.01):
    originals = []
    for i in layer_indices:
        layer = model.model.layers[i]
        orig = layer.self_attn
        wrapper = wrapper_class(orig, epsilon=epsilon).cuda()
        layer.self_attn = wrapper
        originals.append((i, orig))
    return originals


def restore_layers(model, originals):
    for i, orig in originals:
        model.model.layers[i].self_attn = orig


def main():
    print("=" * 70)
    print("  ROW-STOCHASTIC vs DOUBLY-STOCHASTIC vs SOFTMAX")
    print("  Diagnostic: is Sinkhorn the root cause of PPL degradation?")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda", attn_implementation="eager",
    )
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    text = load_wikitext_test_text(max_chars=MAX_CHARS)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    n_tokens = input_ids.shape[1]
    n_windows = (n_tokens - WINDOW) // STRIDE
    print(f"  {n_tokens} tokens, {n_windows} windows")

    results = {}

    # 1. Baseline
    print("\n--- BASELINE (softmax) ---")
    t0 = time.time()
    ppl_base = eval_perplexity(model, input_ids)
    dt = time.time() - t0
    print(f"  PPL = {ppl_base:.3f} ({dt:.1f}s)")
    results["baseline"] = ppl_base

    # 2. Row-stochastic SIRI (no Sinkhorn) on safe layers, eps=0.01
    for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
        print(f"\n--- ROW-STOCHASTIC SIRI eps={eps} (safe layers) ---")
        originals = swap_layers(model, RowStochasticSIRI, SAFE_LAYERS, epsilon=eps)
        t0 = time.time()
        try:
            ppl = eval_perplexity(model, input_ids)
            delta = (ppl - ppl_base) / ppl_base * 100
            gate = "PASS" if delta <= 2.0 else "FAIL"
            print(f"  PPL = {ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
        except Exception as e:
            ppl = float('nan')
            delta = float('nan')
            gate = "ERROR"
            print(f"  ERROR: {e}")
        restore_layers(model, originals)
        results[f"row_stoch_eps{eps}"] = {"ppl": ppl, "delta": delta, "gate": gate}

    # 3. Row-stochastic SIRI on ALL layers, eps=0.1 (least destructive per Phase 5)
    print(f"\n--- ROW-STOCHASTIC SIRI eps=0.1 (ALL 28 layers) ---")
    originals = swap_layers(model, RowStochasticSIRI, range(28), epsilon=0.1)
    t0 = time.time()
    try:
        ppl = eval_perplexity(model, input_ids)
        delta = (ppl - ppl_base) / ppl_base * 100
        gate = "PASS" if delta <= 2.0 else "FAIL"
        print(f"  PPL = {ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
    except Exception as e:
        ppl = float('nan')
        delta = float('nan')
        gate = "ERROR"
        print(f"  ERROR: {e}")
    restore_layers(model, originals)
    results["row_stoch_eps0.1_all"] = {"ppl": ppl, "delta": delta, "gate": gate}

    # 4. Softmax baseline (re-swap to validate reproducibility)
    print(f"\n--- SOFTMAX BASELINE (re-test) ---")
    originals = swap_layers(model, SoftmaxBaseline, SAFE_LAYERS, epsilon=0.0)
    t0 = time.time()
    try:
        ppl = eval_perplexity(model, input_ids)
        delta = (ppl - ppl_base) / ppl_base * 100
        gate = "PASS" if delta <= 2.0 else "FAIL"
        print(f"  PPL = {ppl:.3f} (Delta={delta:+.2f}%) [{gate}] ({time.time()-t0:.1f}s)")
    except Exception as e:
        ppl = float('nan')
        delta = float('nan')
        gate = "ERROR"
        print(f"  ERROR: {e}")
    restore_layers(model, originals)
    results["softmax_retest"] = {"ppl": ppl, "delta": delta, "gate": gate}

    # Save
    out_dir = Path("results_real/perplexity_bt_v5_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "row_stochastic_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Baseline PPL: {ppl_base:.3f}")
    print("-" * 70)
    for k, v in results.items():
        if k == "baseline":
            continue
        ppl_str = f"{v['ppl']:.3f}" if v['ppl'] == v['ppl'] else "NaN"
        delta_str = f"{v['delta']:+.2f}%" if v['delta'] == v['delta'] else "NaN"
        print(f"  {k:<30} PPL={ppl_str:>10}  Delta={delta_str:>10}  [{v['gate']}]")
    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
