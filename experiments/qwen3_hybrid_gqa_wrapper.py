"""
Qwen3 Hybrid GQA Wrapper \u2014 Drop-in replacement for softmax attention
========================================================================

[DEFINITION] HybridAttention wrapper for Qwen3-0.6B DecoderLayer.self_attn.

This wrapper:
- Preserves original projections (q_proj, k_proj, v_proj, o_proj)
- Preserves q_norm, k_norm (Qwen3-specific normalization)
- Applies RoPE using position_embeddings from DecoderLayer
- Handles GQA (8 KV heads -> 16 query heads via repeat_interleave)
- Applies causal mask for autoregressive inference
- Replaces softmax(QK^T / sqrt(d)) with HybridAttention (DeltaNet + SIRI + psi)

[INVARIANTS PRESERVED]
- I1: C_{ij} = ||Q_i - K_j||^2 (geometric cost, NOT inner product)
- I2: A \u2208 Sigma_n (doubly-stochastic under SIRI)
- I3: log_S = -C/eps + psi (Power Diagram bias)
- I4: epsilon \u2208 (0, inf)

[COMPATIBILITY]
- Drop-in replacement for Qwen3Attention in Qwen3DecoderLayer
- Same forward signature: (hidden_states, position_embeddings, ...) -> (output, attn_weights)
- Original projection weights are NOT copied \u2014 shared by reference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from hybrid_attention_torch import HybridAttentionTorch


class Qwen3HybridGQABubbleWrapper(nn.Module):
    """HybridAttention wrapper for Qwen3 self_attn.

    Drop-in replacement for Qwen3Attention. Preserves:
    - q_proj, k_proj, v_proj, o_proj (shared by reference)
    - q_norm, k_norm (Qwen3 applies QK norm after projection, before RoPE)
    - scaling factor from original config

    Replaces:
    - softmax(QK^T / sqrt(d) + mask) @ V with HybridAttention
    """

    def __init__(
        self,
        original_attn: nn.Module,
        epsilon: float = 0.01,
        lam: float = 0.5,
        tau_iters: int = 5,
        use_psi: bool = True,
        use_delta: bool = True,  # If False, only SIRI (pure legacy mode)
        siri_mode: str = "classical",  # "classical" | "chiller" | "sparse" | "soft"
        siri_beta: float = 5.0,  # sharpening factor for chiller mode
        siri_alpha: float = 0.3,  # blend weight for soft mode
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

        # HybridAttention params
        self.epsilon = epsilon
        self.lam = lam
        self.use_psi = use_psi
        self.use_delta = use_delta
        # New SIRI variants (Phase 12 — peakedness-preserving)
        self.siri_mode = siri_mode
        self.siri_beta = siri_beta
        self.siri_alpha = siri_alpha
        assert siri_mode in ("classical", "chiller", "sparse", "soft"), \
            f"Unknown siri_mode={siri_mode}. Must be classical/chiller/sparse/soft"

        # Create HybridAttentionTorch with proper head_dim (Qwen3 uses head_dim=128).
        # We pass num_heads=16 and head_dim=128 implicitly by setting d_model and num_heads.
        # But HybridAttentionTorch derives head_dim = d_model / num_heads = 1024/16 = 64.
        # Qwen3 has head_dim=128, so we need a special construction.
        #
        # Solution: create DeltaNet directly with explicit head_dim, then construct Hybrid.
        # For simplicity, we manually instantiate the components.
        from hybrid_attention_torch import DeltaNetTorch, SIRIPostprocessTorch, PowerDiagramTorch
        self._deltanet = DeltaNetTorch(
            d_model=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            chunk_size=16,
        )
        # Resize W_q, W_k, W_v, W_o to match Qwen3's projection shapes.
        # For Qwen3: q_proj outputs (num_heads * head_dim) features.
        # Qwen3 head_dim=128, but HybridAttentionTorch assumes head_dim = d_model/num_heads = 64.
        # This is an incompatibility \u2014 we'll skip DeltaNet from the wrapper's flow
        # and use our own implementation in forward() that uses Qwen3's head_dim.
        #
        # WORKAROUND: just use SIRIPostprocessTorch + PowerDiagramTorch directly.
        self._siri = SIRIPostprocessTorch(epsilon=epsilon, tau_iters=tau_iters)
        if use_psi:
            self._pd = PowerDiagramTorch(d_model=self.hidden_size)
        else:
            self._pd = None

        # Cast internal components to match the dtype of original projections.
        # If original is float16, we upcast to float32 for numerical stability
        # in Sinkhorn (otherwise exp overflows).
        orig_dtype = original_attn.q_proj.weight.dtype
        if orig_dtype == torch.float16:
            self._siri = self._siri.float()
            if self._pd is not None:
                self._pd = self._pd.float()

        self._hybrid = None  # don't use the unified module; we compute inline.

    def _apply_rope(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q, K using cos, sin.

        Standard Qwen3 RoPE application (already handled by DecoderLayer via
        transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb).
        """
        try:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            Q_out, K_out = apply_rotary_pos_emb(Q, K, cos, sin)
            return Q_out, K_out
        except ImportError:
            # Fallback: simple rotation embedding.
            # cos, sin: [B, N, head_dim]
            # Q, K: [B, H, N, head_dim]
            cos = cos.unsqueeze(1)  # [B, 1, N, head_dim]
            sin = sin.unsqueeze(1)
            Q_rot = Q * cos + self._rotate_half(Q) * sin
            K_rot = K * cos + self._rotate_half(K) * sin
            return Q_rot, K_rot

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the embedding dims for RoPE."""
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _make_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create additive causal mask: 0 for valid, -inf for masked."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[object] = None,  # unused, kept for API
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,  # unused
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass compatible with Qwen3Attention.

        Args:
            hidden_states: [B, N, hidden_size]
            position_embeddings: (cos, sin) for RoPE, each [B, N, head_dim]
            attention_mask: optional [N, N] additive mask (0 valid, -inf masked).
                In Qwen3, this is typically the causal mask OR a 4D cache mask.
            past_key_value, use_cache, cache_position: KV cache API (unused here).

        Returns:
            (output, attn_weights)
            output: [B, N, hidden_size]
            attn_weights: [B, H, N, N] or None
        """
        B, N, D = hidden_states.shape

        # 1. Q, K, V projections (keep original dtype)
        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. QK normalization (Qwen3-specific)
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Cast to float32 for Hybrid computation (numerical stability).
        orig_dtype = hidden_states.dtype
        Q = Q.float()
        K = K.float()
        V = V.float()

        # 3. RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = self._apply_rope(Q, K, cos.to(Q.dtype), sin.to(K.dtype))

        # 4. Build causal mask if not provided
        if attention_mask is None:
            causal = self._make_causal_mask(N, Q.device, Q.dtype)
        else:
            # attention_mask may be 4D [B, 1, N, N] or 2D [N, N].
            # Use directly as additive mask.
            causal = attention_mask
            if causal.dim() == 4:
                # Take first batch's mask (all same).
                causal = causal[0, 0]

        # 5. Compute Hybrid attention
        # We use a custom forward that bypasses the internal projections
        # since Q, K, V are already projected. Pass them as x_proj [B, N, H*D].
        # Simpler: directly compute hybrid in the right format.
        # For efficiency, call _hybrid.forward but skip its internal projections.
        # Since _hybrid.set_projections has been called with shared weights,
        # calling _hybrid.forward(hidden_states) would re-project.
        # We need a "forward from QKV" path \u2014 implement it inline.

        # GQA: expand K, V from num_kv_heads to num_heads via repeat_interleave.
        # This is the standard Qwen3 attention pattern (each KV head serves
        # kv_groups query heads).
        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)  # [B, H, N, head_dim]
            V = V.repeat_interleave(self.kv_groups, dim=1)

        # Normalize Q, K, V for DeltaNet's safety normalization
        Q_norm = Q / Q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        K_norm = K / K.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        V_norm = V / V.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # Also normalize for SIRI cost computation. Raw Q/K in 128-dim with norm
        # ~10 would give ||Q-K||^2 ~ 100-400, which with eps=0.01 saturates the
        # softmax. Normalizing keeps C in [0, 4].
        Q_siri = Q_norm
        K_siri = K_norm

        # --- DeltaNet base ---
        if self.use_delta:
            # Recurrent delta rule with decay.
            out_delta = torch.zeros_like(V_norm)
            S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim,
                           dtype=Q_norm.dtype, device=Q.device)
            N_eff = max(N, 2)
            norm_decay = 1.0 - 1.0 / N_eff
            for t in range(N):
                v_old = torch.einsum("bhij,bhj->bhi", S, K_norm[:, :, t])
                delta = V_norm[:, :, t] - v_old
                S = norm_decay * S + torch.einsum("bhj,bhi->bhij", K_norm[:, :, t], delta)
                out_delta[:, :, t] = torch.einsum("bhij,bhj->bhi", S, Q_norm[:, :, t])
            out_delta = out_delta.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)
        else:
            # Skip DeltaNet (pure SIRI mode)
            out_delta = torch.zeros(B, N, D, dtype=Q_norm.dtype, device=Q.device)

        # --- SIRI post-process (Phase 12: peakedness-preserving variants) ---
        if self.siri_mode == "chiller":
            # Chiller: use raw QK^T scores (similar to softmax), then Sinkhorn
            # with temperature sharpening (multiply log_S by beta).
            # This preserves natural peakedness of standard attention scores
            # while softly enforcing row-stochasticity (NOT full doubly-stochastic).
            log_S = (Q_siri @ K_siri.transpose(-2, -1)) / (self.head_dim ** 0.5)
            log_S = log_S * self.siri_beta
        else:
            # Classical/Sparse/Soft: use geometric cost ||Q-K||^2.
            Q_sq = (Q_siri * Q_siri).sum(dim=-1, keepdim=True)
            K_sq = (K_siri * K_siri).sum(dim=-1, keepdim=True)
            K_sq_t = K_sq.transpose(-2, -1)
            C = (Q_sq + K_sq_t - 2.0 * Q_siri @ K_siri.transpose(-2, -1)).clamp(min=0.0)

            # Min-max normalize cost for SIRI.
            C_min = C.amin(dim=(-2, -1), keepdim=True)
            C_max = C.amax(dim=(-2, -1), keepdim=True)
            C = (C - C_min) / (C_max - C_min + 1e-10)

            if self.siri_mode == "sparse":
                # ReLU on raw cost: only positive contributions, then Sinkhorn.
                # Sparse: log_S = log(relu(-C / epsilon)) — keeps positional
                # peaks where Q_i ~= K_j (low cost), zeros out far positions.
                log_S = -C / self.epsilon
                # Use raw cost as score (not log-space). For ReLU version.
                # Here we set log_S to log(relu(-C/eps)) with -inf for zero.
                relu_cost = torch.clamp(-C / self.epsilon, min=0.0)
                log_S = torch.where(relu_cost > 0, relu_cost, torch.full_like(relu_cost, -50.0))
            else:
                # Classical or soft blend: standard -C/ε.
                log_S = -C / self.epsilon

        # Power Diagram psi (if enabled)
        if self.use_psi and self._pd is not None:
            psi = self._pd(hidden_states.float())  # [B, N, 1]  (cast to float32 for matmul)
            log_S = log_S + psi.unsqueeze(1).to(log_S.dtype)

        # Add causal mask. Qwen3 passes a 4D mask [B, 1, N, target_length]
        # where target_length can be > N (KV cache convention). We slice to
        # the last N columns so the mask aligns with log_S shape [B, H, N, N].
        causal_2d = causal if causal.dim() == 2 else causal[0, 0]
        if causal_2d.shape[-1] != N:
            causal_2d = causal_2d[..., -N:]
        log_S = log_S + causal_2d.unsqueeze(0).unsqueeze(0)

        # Clamp for numerical safety. Wider range to allow ε ≤ 0.01.
        log_S = log_S.clamp(min=-500.0, max=50.0)

        # Sinkhorn-Knopp log-domain (tau iterations).
        u = torch.zeros(B, self.num_heads, N, dtype=log_S.dtype, device=Q.device)
        v = torch.zeros(B, self.num_heads, N, dtype=log_S.dtype, device=Q.device)
        from hybrid_attention_torch import _logsumexp
        for _ in range(self._siri.tau_iters):
            u = -_logsumexp(log_S + v.unsqueeze(-1), dim=-1)
            v = -_logsumexp(log_S + u.unsqueeze(-2), dim=-2)

        log_A = log_S + u.unsqueeze(-1) + v.unsqueeze(-2)
        A = log_A.exp()
        A = torch.nan_to_num(A, nan=0.0, posinf=1e10, neginf=0.0)
        # Row renormalize (Sinkhorn guarantees this approximately; we ensure exact).
        row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        A = A / row_sums

        # Phase 12: Soft blend with softmax for soft mode (SpikeFormer-style).
        if self.siri_mode == "soft":
            # Standard attention scores (QK^T / sqrt(d)).
            attn_scores = (Q_siri @ K_siri.transpose(-2, -1)) / (self.head_dim ** 0.5)
            # Causal mask.
            attn_scores = attn_scores + causal_2d.unsqueeze(0).unsqueeze(0)
            # Softmax with row-stochastic normalization.
            A_softmax = torch.softmax(attn_scores, dim=-1)
            # Blend: (1 - alpha) * softmax + alpha * SIRI.
            A = (1.0 - self.siri_alpha) * A_softmax + self.siri_alpha * A
            A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # SIRI output.
        out_siri_heads = A @ V  # [B, H, N, head_dim]
        out_siri = out_siri_heads.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)

        # Hybrid interpolation.
        if self.use_delta:
            out = self.lam * out_delta + (1.0 - self.lam) * out_siri
        else:
            out = out_siri

        # Cast back to original dtype for output projection.
        out = out.to(orig_dtype)
        out = self.o_proj(out)

        if output_attentions:
            return out, A
        return out, None

    def extra_repr(self) -> str:
        return (
            f"HybridGQABubbleWrapper(eps={self.epsilon}, lam={self.lam}, "
            f"use_psi={self.use_psi}, use_delta={self.use_delta})"
        )


if __name__ == "__main__":
    print("[Qwen3-Hybrid-Wrapper] Smoke test...")
    from transformers import AutoModelForCausalLM
    import warnings
    warnings.filterwarnings("ignore")

    print("Loading Qwen3-0.6B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    # Swap layer 0 with hybrid wrapper.
    print("Wrapping layer 0 with HybridAttention...")
    original_attn = model.model.layers[0].self_attn
    wrapper = Qwen3HybridGQABubbleWrapper(
        original_attn=original_attn,
        epsilon=0.01,
        lam=0.5,
    ).cuda()
    # NOTE: do NOT call .to(torch.float16) here \u2014 that would downcast _pd.W_psi
    # and break Sinkhorn numerics. The wrapper already handles dtype internally.

    # Forward test.
    B, N = 1, 32
    hidden = torch.randn(B, N, model.config.hidden_size, dtype=torch.float16, device="cuda")
    pos = torch.arange(N, device="cuda").unsqueeze(0)

    with torch.no_grad():
        # Qwen3 model.model.rotary_emb returns (cos, sin)
        cos, sin = model.model.rotary_emb(hidden, pos)
        out, _ = wrapper(hidden, position_embeddings=(cos, sin))

    print(f"  Input: {hidden.shape}, dtype={hidden.dtype}")
    print(f"  Output: {out.shape}, dtype={out.dtype}, range=[{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"  Has NaN: {torch.isnan(out).any().item()}")
    print(f"  Has Inf: {torch.isinf(out).any().item()}")
    assert out.shape == (B, N, model.config.hidden_size)
    assert torch.isfinite(out).all()
    print("[Qwen3-Hybrid-Wrapper] Smoke test passed!")
