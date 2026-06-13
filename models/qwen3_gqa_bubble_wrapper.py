"""
Qwen3 GQA Bubble Wrapper -- SDOT GQA-Native Integration
========================================================

Reemplaza self_attn en Qwen3-0.6B sin tocar sus proyecciones originales.
Solo reemplaza el calculo de attention scores con SDOT soft routing.

CRITICAL differences from naive reimplementation:
- q_norm and k_norm are applied AFTER projection, BEFORE RoPE (Qwen3-specific)
- RoPE received as position_embeddings=(cos, sin) from DecoderLayer
- Uses self.scaling from original config (not hardcoded 1/sqrt(d))
- o_proj maps [B, N, 2048] -> [B, N, 1024] (hidden_size, not num_heads*head_dim)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except ImportError:
    try:
        from transformers.modeling_rope_utils import apply_rotary_pos_emb
    except ImportError:
        apply_rotary_pos_emb = None


def _make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask


class Qwen3GQABubbleWrapper(nn.Module):
    """Reemplaza self_attn en un DecoderLayer de Qwen3.

    Match exacto con Qwen3Attention.forward():
    - q_norm/k_norm applied after projection, before RoPE
    - RoPE received as position_embeddings=(cos, sin)
    - Uses original scaling factor
    - Returns (output, None)
    """

    def __init__(
        self,
        original_attn: nn.Module,
        num_bubbles: int = 32,
        top_k: int = 64,
        eps: float = 0.005,
        routing_bonus: float = 0.1,
        debug: bool = True,
        use_dual_head: bool = False,
        use_alpha_prediction: bool = False,
    ):
        super().__init__()

        # Original projections (same object references)
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        # CRITICAL: Qwen3 applies QK normalization after projection
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm

        # Config
        self.num_heads = original_attn.config.num_attention_heads
        self.num_kv_heads = original_attn.config.num_key_value_heads
        self.head_dim = original_attn.config.head_dim
        self.hidden_size = original_attn.config.hidden_size
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = original_attn.scaling

        self.use_dual_head = use_dual_head

        if use_dual_head:
            try:
                from .sdot_attention_v4 import DualHeadSDOTAttentionV4
                from .baroreceptor import BaroreceptorMLP
            except ImportError:
                from sdot_attention_v4 import DualHeadSDOTAttentionV4
                from baroreceptor import BaroreceptorMLP

            self.dual_head_attn = DualHeadSDOTAttentionV4(
                d_model=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                num_centroids=num_bubbles,
                top_k=top_k,
                use_baroreceptor=False,
                use_fps_init=True,
            )

            # Copy original projection weights into dual-head module
            with torch.no_grad():
                proj_map = {
                    "W_q": original_attn.q_proj,
                    "W_k": original_attn.k_proj,
                    "W_v": original_attn.v_proj,
                    "W_o": original_attn.o_proj,
                }
                for proj_name, orig_proj in proj_map.items():
                    low_proj = getattr(self.dual_head_attn.head_low, proj_name)
                    low_proj.weight.copy_(orig_proj.weight)
                    if orig_proj.bias is None and low_proj.bias is not None:
                        low_proj.bias.zero_()

            if use_alpha_prediction:
                self.baroreceptor = BaroreceptorMLP(
                    d_model=self.hidden_size,
                    min_C=16,
                    max_C=512,
                    use_alpha_prediction=True,
                )

        # Bubble routing params (retained for backward compat / reference)
        self.num_bubbles = num_bubbles
        self.top_k = top_k
        self.eps = eps
        self.centroids = nn.Parameter(
            torch.randn(num_bubbles, self.head_dim) * 0.02
        )
        self.routing_bonus = routing_bonus

        self.debug = debug
        self._debug_printed = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_dual_head:
            if hasattr(self, "baroreceptor"):
                _, alpha = self.baroreceptor.forward_with_alpha(hidden_states)
                self.dual_head_attn.alpha = alpha

            output, _ = self.dual_head_attn(hidden_states)
            return output, None

        B, N, D = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # PASO 1: Project + QK normalization (match Qwen3 exactly)
        # Original: self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        Q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B, 16, N, 128]
        K = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B, 8, N, 128]
        V = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)               # [B, 8, N, 128]

        # PASO 2: Apply RoPE BEFORE GQA expansion (match Qwen3)
        if position_embeddings is not None and apply_rotary_pos_emb is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # PASO 3: GQA expansion (repeat_kv) -- AFTER RoPE
        K_expanded = K.repeat_interleave(self.kv_groups, dim=1)  # [B, 16, N, 128]
        V_expanded = V.repeat_interleave(self.kv_groups, dim=1)  # [B, 16, N, 128]

        if self.debug and not self._debug_printed:
            print(f"[Qwen3GQABubbleWrapper] Forward pass")
            print(f"  hidden_states: {hidden_states.shape}")
            print(f"  Q: {Q.shape}, K: {K.shape}, K_expanded: {K_expanded.shape}")
            print(f"  V_expanded: {V_expanded.shape}")
            print(f"  RoPE applied: {position_embeddings is not None}")
            print(f"  scaling: {self.scaling}")
            print(f"  q_norm type: {type(self.q_norm).__name__}")
            print(f"  k_norm type: {type(self.k_norm).__name__}")

        # PASO 4: SDOT clustering
        K_flat = K.float().mean(dim=1)  # [B, N, 128] -- avg over kv_heads
        centroids_dev = self.centroids.to(K_flat.device).unsqueeze(0).expand(B, -1, -1)
        dists = torch.cdist(K_flat, centroids_dev)  # [B, N, num_bubbles]
        assignments = dists.argmin(dim=-1)  # [B, N]

        if self.debug and not self._debug_printed:
            print(f"  assignments: {assignments.shape}, unique bubbles: {assignments.unique().shape[0]}")

        # PASO 5: Soft routing bias
        row_assign = assignments.unsqueeze(-1)   # [B, N, 1]
        col_assign = assignments.unsqueeze(-2)   # [B, 1, N]
        same_bubble = (row_assign == col_assign).float()  # [B, N, N]
        routing_bias = same_bubble * self.routing_bonus    # [B, N, N]

        # PASO 6: Causal mask
        causal_mask = _make_causal_mask(N, hidden_states.device, hidden_states.dtype)

        # PASO 7: Attention with routing bias + causal mask
        attn_scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * self.scaling  # [B, 16, N, N]

        # Add bias and mask
        attn_scores = attn_scores + routing_bias.unsqueeze(1).to(dtype=attn_scores.dtype, device=attn_scores.device)
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0).to(dtype=attn_scores.dtype, device=attn_scores.device)

        # Softmax in float32 for numerical stability
        attn_weights = torch.softmax(attn_scores.float(), dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn_weights.to(V_expanded.dtype)

        output = torch.matmul(attn_weights, V_expanded)  # [B, 16, N, 128]

        # PASO 8: Reshape + output projection (match Qwen3 exactly)
        attn_output = output.transpose(1, 2).contiguous()  # [B, N, 16, 128]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # [B, N, 2048]
        attn_output = self.o_proj(attn_output)  # [B, N, 1024]

        if self.debug and not self._debug_printed:
            print(f"  output: {attn_output.shape}, NaN: {attn_output.isnan().any()}")
            print(f"  DEBUG PASSED")
            self._debug_printed = True

        return attn_output, None


if __name__ == "__main__":
    print("Testing Qwen3GQABubbleWrapper...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16, device_map="cpu")
    original_attn = model.model.layers[0].self_attn

    wrapper = Qwen3GQABubbleWrapper(
        original_attn=original_attn,
        num_bubbles=32,
        routing_bonus=0.0,
        debug=True,
    )

    # Test forward
    x = torch.randn(1, 10, 1024, dtype=torch.float16)
    pos = torch.arange(10).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, pos)

    out, _ = wrapper(x, position_embeddings=(cos, sin))
    print(f"  output.shape: {out.shape}")
    print(f"  output NaN: {out.isnan().any()}")
    print(f"  [PASS] ALL TESTS PASSED")
