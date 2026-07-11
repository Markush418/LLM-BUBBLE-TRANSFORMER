"""
Qwen3 Focus Bubble Wrapper — Drop-in replacement for softmax attention
======================================================================

[DEFINITION] FocusBubbleAttention wrapper for Qwen3-0.6B DecoderLayer.self_attn.

This wrapper:
- Preserves original projections (q_proj, k_proj, v_proj, o_proj)
- Preserves q_norm, k_norm (Qwen3-specific normalization)
- Applies RoPE using position_embeddings from DecoderLayer
- Handles GQA (8 KV heads -> 16 query heads via repeat_interleave)
- Applies causal mask for autoregressive inference
- Replaces softmax(QK^T / sqrt(d)) with FocusBubbleAttention

Key insight from Focus (arXiv:2604.03260):
  - Sinkhorn can IMPROVE PPL when used for grouping
  - Softmax within groups preserves peakedness
  - Works as retrofit on frozen models
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from focus_bubble_attention import FocusBubbleAttention, FocusBubbleDeltaNet


class Qwen3FocusBubbleWrapper(nn.Module):
    """FocusBubbleAttention wrapper for Qwen3 self_attn.

    Drop-in replacement for Qwen3Attention. Preserves:
    - q_proj, k_proj, v_proj, o_proj (shared by reference)
    - q_norm, k_norm (Qwen3 applies QK norm after projection, before RoPE)
    - scaling factor from original config

    Replaces:
    - softmax(QK^T / sqrt(d) + mask) @ V with FocusBubbleAttention
    """

    def __init__(
        self,
        original_attn: nn.Module,
        epsilon: float = 0.1,
        tau_iters: int = 5,
        use_psi: bool = True,
        use_delta: bool = False,  # If True, combine with DeltaNet
        lam: float = 0.5,  # interpolation weight for DeltaNet mode
        dropout: float = 0.0,
    ):
        super().__init__()
        # Original projections (same object references)
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        # Qwen3 applies q_norm/k_norm after projection, before RoPE
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm

        # Config from original
        self.num_heads = original_attn.config.num_attention_heads
        self.num_kv_heads = original_attn.config.num_key_value_heads
        self.head_dim = original_attn.config.head_dim
        self.hidden_size = original_attn.config.hidden_size
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = original_attn.scaling

        # Focus Bubble parameters
        self.epsilon = epsilon
        self.tau_iters = tau_iters
        self.use_psi = use_psi
        self.use_delta = use_delta
        self.lam = lam

        # Create FocusBubbleAttention or FocusBubbleDeltaNet
        if use_delta:
            self._focus_attn = FocusBubbleDeltaNet(
                d_model=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                epsilon=epsilon,
                tau_iters=tau_iters,
                lam=lam,
                use_psi=use_psi,
                dropout=dropout,
            )
        else:
            self._focus_attn = FocusBubbleAttention(
                d_model=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                epsilon=epsilon,
                tau_iters=tau_iters,
                use_psi=use_psi,
                dropout=dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """Forward pass compatible with Qwen3Attention.

        Args:
            hidden_states: [B, N, D]
            position_embeddings: (cos, sin) from rotary_emb — NOT used by FocusBubble
            attention_mask: [B, 1, N, N] or [B, N, N] additive causal mask
            past_key_value: ignored (no cache in this implementation)
            output_attentions: if True, return attention weights

        Returns:
            output: [B, N, D]
            attn_weights: [B, H, N, N] or None
            past_key_value: None
        """
        orig_dtype = hidden_states.dtype
        B, N, _ = hidden_states.shape

        # Project Q, K, V using original projections
        # Qwen3: view + transpose BEFORE q_norm/k_norm (head-wise RMSNorm over D_h)
        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Qwen3: apply q_norm, k_norm AFTER reshape (normalizes over head_dim)
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Apply RoPE if position_embeddings provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            Q, K = apply_rotary_pos_emb(Q, K, cos.to(Q.dtype), sin.to(K.dtype))

        # GQA: expand K/V to match Q heads
        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)
            V = V.repeat_interleave(self.kv_groups, dim=1)

        # Cast to float32 for numerical stability (SIRI computation)
        orig_dtype = hidden_states.dtype
        Q = Q.float()
        K = K.float()
        V = V.float()

        # Build causal 2D mask [N, N] from attention_mask [B, 1, N, N]
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                # [B, 1, N, target_length] — Qwen3 KV-cache convention
                target_length = attention_mask.shape[-1]
                causal_2d = attention_mask[0, 0, :, -N:]  # [N, target_length]
                if causal_2d.shape[-1] != N:
                    causal_2d = causal_2d[:, -N:]
            else:
                causal_2d = attention_mask
        else:
            causal_2d = torch.triu(
                torch.full((N, N), float("-inf"), device=hidden_states.device, dtype=torch.float32),
                diagonal=1,
            )

        # Apply RoPE to Q/K for SIRI cost computation
        # Then compute SIRI attention
        # FocusBubbleAttention expects unprojected Q, K, V in [B, N, D] format
        # But we have [B, H, N, D_h] format after RoPE
        # So we need to adapt

        # For FocusBubbleAttention, we pass through the module directly
        # The module handles projections internally, but we've already applied RoPE
        # So we need to use a different approach: compute attention manually

        # Step 1: Compute dot-product scores
        log_S = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        # Step 2: Add Power Diagram bias
        if self.use_psi and hasattr(self._focus_attn, 'psi') and self._focus_attn.psi is not None:
            psi = self._focus_attn.psi
            log_S = log_S + psi.unsqueeze(0).unsqueeze(-1)

        # Step 3: Sinkhorn for grouping
        # Apply causal mask
        log_S = log_S + causal_2d.unsqueeze(0).unsqueeze(0)

        # Temperature scaling
        log_S_scaled = log_S * self.epsilon

        # Clamp for numerical stability
        log_S_scaled = log_S_scaled.clamp(min=-50.0, max=50.0)

        # Sinkhorn iterations
        for _ in range(self.tau_iters):
            log_S_scaled = log_S_scaled - torch.logsumexp(log_S_scaled, dim=-1, keepdim=True)
            log_S_scaled = log_S_scaled - torch.logsumexp(log_S_scaled, dim=-2, keepdim=True)

        # Convert to doubly-stochastic grouping
        groups = torch.exp(log_S_scaled)

        # Step 4: Standard softmax on grouped scores (preserves peakedness!)
        attn_scores = log_S + torch.log(groups + 1e-10)

        # Standard softmax (preserves peakedness!)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Step 5: Compute output
        out_siri = torch.matmul(attn_weights, V)

        # If using DeltaNet mode, also compute DeltaNet output
        if self.use_delta and hasattr(self._focus_attn, '_deltanet_forward'):
            out_delta = self._focus_attn._deltanet_forward(Q, K, V, causal_2d)
            out_siri = self.lam * out_delta + (1 - self.lam) * out_siri

        # Reshape to [B, N, D]
        out_siri = out_siri.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)

        # Output projection
        out = out_siri.to(orig_dtype)
        out = self.o_proj(out)

        if output_attentions:
            return out, attn_weights
        return out, None


# =============================================================================
# Test
# =============================================================================


if __name__ == "__main__":
    print("Qwen3FocusBubbleWrapper module loaded successfully.")
    print("Use with Qwen3DecoderLayer.self_attn for drop-in replacement.")
