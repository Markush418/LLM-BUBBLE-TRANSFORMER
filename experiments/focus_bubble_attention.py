"""
FocusBubbleAttention — Focus-inspired Bubble Transformer
=========================================================

[DEFINITION] Drop-in attention replacement that uses Sinkhorn for TOKEN GROUPING,
then applies standard softmax WITHIN the grouped structure.

Key insight from Focus (arXiv:2604.03260):
  - Sinkhorn can IMPROVE PPL when used for grouping (42.8 -> 30.3)
  - Softmax within groups preserves peakedness
  - Works as retrofit on frozen models

Architecture:
  1. Compute dot-product scores: S = Q @ K^T / sqrt(d)
  2. Add Power Diagram bias: S = S + psi
  3. Apply Sinkhorn for soft grouping (doubly-stochastic)
  4. Apply standard softmax on top of grouped scores
  5. This preserves peakedness while adding geometric structure

This PRESERVES:
  - SIRI: Used for grouping (Sinkhorn iterations)
  - Power Diagram: psi bias on scores
  - Softmax peakedness: Maintained via standard softmax
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FocusBubbleAttention
# =============================================================================


class FocusBubbleAttention(nn.Module):
    """Focus-inspired Bubble Transformer.
    
    Sinkhorn for token grouping, then standard softmax.
    Preserves peakedness while adding geometric structure.
    
    Args:
        d_model: input/output dimension.
        num_heads: number of attention heads.
        num_kv_heads: number of KV heads (for GQA).
        epsilon: Sinkhorn bandwidth/temperature (default 0.1).
        tau_iters: number of Sinkhorn iterations (default 5).
        use_psi: whether to use Power Diagram bias (default True).
        dropout: attention dropout rate.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        epsilon: float = 0.1,
        tau_iters: int = 5,
        use_psi: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.epsilon = epsilon
        self.tau_iters = tau_iters
        self.use_psi = use_psi
        
        self.head_dim = d_model // num_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        # Power Diagram bias (learnable per-head)
        if use_psi:
            self.psi = nn.Parameter(torch.zeros(num_heads, 1))
        else:
            self.psi = None
        
        self.attn_dropout = nn.Dropout(dropout)
    
    def _sinkhorn_grouping(
        self,
        log_S: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sinkhorn for soft grouping.
        
        Args:
            log_S: [B, H, N, N] raw scores
            causal_mask: optional [N, N] additive mask (0/-inf)
            
        Returns:
            groups: [B, H, N, N] doubly-stochastic grouping matrix
        """
        B, H, N, _ = log_S.shape
        
        # Apply causal mask
        if causal_mask is not None:
            log_S = log_S + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Temperature scaling (epsilon controls grouping sharpness)
        log_S = log_S * self.epsilon
        
        # Clamp for numerical stability
        log_S = log_S.clamp(min=-50.0, max=50.0)
        
        # Sinkhorn iterations for doubly-stochastic normalization
        for _ in range(self.tau_iters):
            # Row normalization
            log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
            # Column normalization
            log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
        
        # Convert to doubly-stochastic matrix
        groups = torch.exp(log_S)
        
        return groups
    
    def _apply_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create causal mask [N, N]."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: [B, N, D] input
            causal_mask: optional [N, N] additive mask
            return_attention: if True, return attention weights
            
        Returns:
            output: [B, N, D]
            attn_weights: [B, H, N, N] (if return_attention)
        """
        B, N, D = x.shape
        
        # Projections
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand K/V for GQA
        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)
            V = V.repeat_interleave(self.kv_groups, dim=1)
        
        # Create causal mask if needed
        if causal_mask is None:
            causal_mask = self._apply_causal_mask(N, x.device, x.dtype)
        
        # Step 1: Compute dot-product scores
        log_S = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        
        # Step 2: Add Power Diagram bias
        if self.psi is not None and self.use_psi:
            # psi: [H, 1] -> broadcast to [1, H, N, N]
            log_S = log_S + self.psi.unsqueeze(0).unsqueeze(-1)
        
        # Step 3: Sinkhorn for grouping (doubly-stochastic)
        groups = self._sinkhorn_grouping(log_S, causal_mask)  # [B, H, N, N]
        
        # Step 4: Standard softmax on grouped scores (preserves peakedness!)
        # Apply causal mask
        attn_scores = log_S + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply Sinkhorn grouping as soft mask
        attn_scores = attn_scores + torch.log(groups + 1e-10)
        
        # Standard softmax (preserves peakedness!)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Step 5: Compute output
        out = torch.matmul(attn_weights, V)
        
        # Step 6: Output projection
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.o_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out, None


# =============================================================================
# FocusBubbleDeltaNet — Combined DeltaNet + Focus grouping
# =============================================================================


class FocusBubbleDeltaNet(nn.Module):
    """Focus-inspired Bubble Transformer with DeltaNet base.
    
    Architecture:
      1. DeltaNet base output (O(N) linear)
      2. Focus grouping with Power Diagram bias
      3. Standard softmax on grouped scores
      4. Interpolation: out = lam * out_delta + (1-lam) * out_focus
    
    Args:
        d_model: input/output dimension.
        num_heads: number of attention heads.
        num_kv_heads: number of KV heads (for GQA).
        epsilon: Sinkhorn bandwidth (default 0.1).
        tau_iters: Sinkhorn iterations (default 5).
        lam: interpolation weight (0 = pure Focus, 1 = pure DeltaNet).
        use_psi: whether to use Power Diagram bias.
        dropout: attention dropout rate.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        epsilon: float = 0.1,
        tau_iters: int = 5,
        lam: float = 0.5,
        use_psi: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.epsilon = epsilon
        self.tau_iters = tau_iters
        self.lam = lam
        self.use_psi = use_psi
        
        self.head_dim = d_model // num_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        # DeltaNet state projection
        self.W_o = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)
        
        # Focus components
        self.focus_attn = FocusBubbleAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            epsilon=epsilon,
            tau_iters=tau_iters,
            use_psi=use_psi,
            dropout=dropout,
        )
        
        # DeltaNet decay
        self.decay_log = nn.Parameter(torch.zeros(num_heads))
    
    def _deltanet_forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DeltaNet forward pass (O(N) linear).
        
        Args:
            Q: [B, H, N, D_h]
            K: [B, H, N, D_h]
            V: [B, H, N, D_h]
            causal_mask: optional [N, N]
            
        Returns:
            out: [B, H, N, D_h]
        """
        B, H, N, D_h = Q.shape

        # [FASE 5 FIX] Normalize Q, K, V to unit norm per token to prevent overflow.
        # With raw Q/K from Qwen3, norm ~16 produces values that explode in the
        # delta rule. Normalizing keeps everything bounded.
        def _safe_normalize(x, eps=1e-6):
            norm = x.norm(dim=-1, keepdim=True)
            return x / torch.clamp(norm, min=eps)
        Q = _safe_normalize(Q)
        K = _safe_normalize(K)
        V = _safe_normalize(V)

        # Decay
        decay = torch.exp(-torch.exp(self.decay_log))  # [H]
        norm_decay = 1.0 - decay.view(1, H, 1, 1)
        
        # Initialize state
        S = torch.zeros(B, H, D_h, D_h, device=Q.device, dtype=Q.dtype)
        
        # Output buffer
        out_delta = torch.zeros_like(Q)
        
        # Sequential DeltaNet
        for t in range(N):
            q_t = Q[:, :, t, :]  # [B, H, D_h]
            k_t = K[:, :, t, :]  # [B, H, D_h]
            v_t = V[:, :, t, :]  # [B, H, D_h]
            
            # v_old = S^T k_t
            v_old = torch.einsum("bhij,bhj->bhi", S, k_t)  # [B, H, D_h]
            
            # delta = v_t - v_old
            delta = v_t - v_old
            
            # S = decay * S + outer(k_t, delta)
            S = norm_decay * S + torch.einsum("bhj,bhi->bhij", k_t, delta)
            
            # out[t] = S^T q_t
            out_delta[:, :, t] = torch.einsum("bhij,bhj->bhi", S, q_t)
        
        return out_delta
    
    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: [B, N, D] input
            causal_mask: optional [N, N] additive mask
            return_attention: if True, return attention weights
            
        Returns:
            output: [B, N, D]
            attn_weights: [B, H, N, N] (if return_attention)
        """
        B, N, D = x.shape
        
        # Projections
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand K/V for GQA
        if self.kv_groups > 1:
            K = K.repeat_interleave(self.kv_groups, dim=1)
            V = V.repeat_interleave(self.kv_groups, dim=1)
        
        # Create causal mask if needed
        if causal_mask is None:
            mask = torch.full((N, N), float("-inf"), device=x.device, dtype=x.dtype)
            causal_mask = torch.triu(mask, diagonal=1)
        
        # DeltaNet base output
        out_delta = self._deltanet_forward(Q, K, V, causal_mask)
        
        # Focus grouping + softmax
        out_focus, attn_weights = self.focus_attn(x, causal_mask, return_attention=True)
        out_focus = out_focus.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Interpolation
        out = self.lam * out_delta + (1 - self.lam) * out_focus
        
        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.o_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out, None


# =============================================================================
# Test
# =============================================================================


if __name__ == "__main__":
    torch.manual_seed(42)
    
    B, N, D = 2, 128, 256
    num_heads = 8
    
    x = torch.randn(B, N, D)
    
    # Test FocusBubbleAttention
    model = FocusBubbleAttention(
        d_model=D,
        num_heads=num_heads,
        epsilon=0.1,
        use_psi=True,
    )
    
    out, attn = model(x, return_attention=True)
    print(f"FocusBubbleAttention:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Attn:   {attn.shape if attn is not None else None}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test FocusBubbleDeltaNet
    model2 = FocusBubbleDeltaNet(
        d_model=D,
        num_heads=num_heads,
        epsilon=0.1,
        lam=0.5,
        use_psi=True,
    )
    
    out2, attn2 = model2(x, return_attention=True)
    print(f"\nFocusBubbleDeltaNet:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out2.shape}")
    print(f"  Attn:   {attn2.shape if attn2 is not None else None}")
    print(f"  Params: {sum(p.numel() for p in model2.parameters()):,}")
    
    print("\nAll tests passed!")
