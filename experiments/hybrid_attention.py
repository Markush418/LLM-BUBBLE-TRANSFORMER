"""
Hybrid Attention — DeltaNet + SIRI post-processing + Power Diagram psi
=======================================================================

[DEFINITION] Hybrid Bubble Transformer attention layer:
  1. Base attention: DeltaNet (linear, O(N)) — default.
  2. Power Diagram psi bias: applied via projection on K.
  3. SIRI post-processing: opt-in Sinkhorn refinement for doubly-stochastic A.
  4. Interpolation: out_final = lam * out_delta + (1 - lam) * out_siri.

This module preserves SIRI and Power Diagram from the original Bubble Transformer
while replacing SDOT with DeltaNet as the default base attention.

Pure NumPy. Accepts torch tensors via input conversion.
"""

import numpy as np
from typing import Optional, Tuple

from deltanet_attention import DeltaNetAttention
from siri_postprocess import (
    siri_sinkhorn_log_domain,
    siri_interpolate,
    _to_numpy,
)
from power_diagrams import PowerDiagramModule, compute_psi_from_keys


class HybridAttention:
    """Hybrid attention: DeltaNet (base) + SIRI (post-process) + Power Diagram psi (bias).

    Architecture:
      out_delta = DeltaNet(Q, K, V)                 # linear attention
      log_S = -C(Q,K)/epsilon + psi(K)             # geometric cost + Power Diagram
      A_siri = Sinkhorn_Knopp(log_S, tau=5)        # SIRI doubly-stochastic
      out_siri = A_siri @ V
      out = lam * out_delta + (1-lam) * out_siri    # hybrid interpolation

    Args:
        d_model: input/output dimension.
        num_heads: number of attention heads.
        epsilon: SIRI bandwidth (default 0.1).
        tau_iters: Sinkhorn iterations (default 5).
        lam: SOTA vs SIRI interpolation (1.0 = pure DeltaNet, 0.0 = pure SIRI).
        chunk_size: DeltaNet chunk size.
        seed: random seed.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        epsilon: float = 0.1,
        tau_iters: int = 5,
        lam: float = 0.5,
        chunk_size: int = 16,
        seed: int = 42,
        normalize_costs: bool = True,
    ):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.epsilon = epsilon
        self.tau_iters = tau_iters
        self.lam = lam
        self.normalize_costs = normalize_costs

        # DeltaNet base attention
        self.deltanet = DeltaNetAttention(
            d_model=d_model,
            num_heads=num_heads,
            chunk_size=chunk_size,
            seed=seed,
        )

        # Power Diagram psi projection
        self.pd = PowerDiagramModule(d_model=d_model, seed=seed)

    def forward(
        self,
        x: np.ndarray,
        return_attention: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward pass.

        Args:
            x: [B, N, d_model]
            return_attention: if True, also return hybrid attention matrix.

        Returns:
            output: [B, N, d_model] (same type as input)
            attention: [B, H, N, N] (NumPy)
        """
        _is_torch = hasattr(x, "detach") and hasattr(x, "cpu")
        x_np = _to_numpy(x).astype(np.float32)
        B, N, D = x_np.shape

        # --- DeltaNet base output ---
        out_delta = self.deltanet(x_np)  # [B, N, d_model]

        # --- Compute Q, K for SIRI refinement ---
        Q = (x_np @ self.deltanet.W_q).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x_np @ self.deltanet.W_k).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x_np @ self.deltanet.W_v).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # --- Power Diagram psi bias (apply on original x, before per-head reshape) ---
        psi = self.pd.compute_psi(x_np)  # [B, N, 1] — Power Diagram bias per token

        # --- Geometric cost matrix ---
        Q_sq = np.sum(Q ** 2, axis=-1, keepdims=True)
        K_sq = np.sum(K ** 2, axis=-1, keepdims=True)
        K_sq_t = np.moveaxis(K_sq, -2, -1)
        C = np.maximum(Q_sq + K_sq_t - 2.0 * np.matmul(Q, np.moveaxis(K, -2, -1)), 0.0)

        if self.normalize_costs:
            C_min = np.min(C, axis=(-2, -1), keepdims=True)
            C_max = np.max(C, axis=(-2, -1), keepdims=True)
            C = (C - C_min) / (C_max - C_min + 1e-10)

        log_S = -C / self.epsilon  # [B, H, N, N]

        # Apply psi bias: log_S[b, h, i, j] += psi[b, j, 0]
        log_S = log_S + psi[:, np.newaxis, :, :]

        # --- SIRI Sinkhorn ---
        A_siri = siri_sinkhorn_log_domain(log_S, tau_iters=self.tau_iters)  # [B, H, N, N]

        # --- SIRI output ---
        siri_head_out = np.matmul(A_siri, V)  # [B, H, N, head_dim]
        out_siri = siri_head_out.transpose(0, 2, 1, 3).reshape(B, N, D)

        # --- Hybrid interpolation ---
        out_np = self.lam * out_delta + (1.0 - self.lam) * out_siri

        if return_attention:
            if _is_torch:
                import torch as _torch
                return (_torch.from_numpy(out_np).to(x.dtype if hasattr(x, "dtype") else _torch.float32),
                        _torch.from_numpy(A_siri).to(x.dtype if hasattr(x, "dtype") else _torch.float32))
            return out_np, A_siri
        if _is_torch:
            import torch as _torch
            return _torch.from_numpy(out_np).to(x.dtype if hasattr(x, "dtype") else _torch.float32)
        return out_np

    def __call__(self, x, return_attention: bool = False):
        return self.forward(x, return_attention=return_attention)


if __name__ == "__main__":
    print("[HybridAttention] Running quick test...")
    rng = np.random.RandomState(42)

    B, N, d_model, num_heads = 2, 32, 128, 4

    # Test 1: Pure DeltaNet (lam=1.0)
    attn = HybridAttention(d_model=d_model, num_heads=num_heads, lam=1.0)
    x = rng.randn(B, N, d_model).astype(np.float32)
    output, attn_matrix = attn(x, return_attention=True)
    print(f"  lam=1.0 (pure DeltaNet): output {output.shape}, attn {attn_matrix.shape}")
    assert output.shape == (B, N, d_model)

    # Test 2: Pure SIRI (lam=0.0)
    attn_siri = HybridAttention(d_model=d_model, num_heads=num_heads, lam=0.0)
    output_siri, attn_siri_m = attn_siri(x, return_attention=True)
    row_sums = attn_siri_m.sum(axis=-1)
    print(f"  lam=0.0 (pure SIRI): output {output_siri.shape}, attn row sums mean={row_sums.mean():.4f}")
    assert np.allclose(row_sums, 1.0, atol=5e-2)

    # Test 3: Hybrid (lam=0.5)
    attn_hybrid = HybridAttention(d_model=d_model, num_heads=num_heads, lam=0.5)
    output_hybrid, attn_hybrid_m = attn_hybrid(x, return_attention=True)
    print(f"  lam=0.5 (hybrid): output {output_hybrid.shape}, attn {attn_hybrid_m.shape}")
    assert output_hybrid.shape == (B, N, d_model)

    # Test 4: torch tensor input
    import torch
    x_torch = torch.randn(B, N, d_model)
    output_torch, attn_torch = attn_hybrid(x_torch, return_attention=True)
    print(f"  torch input: output {output_torch.shape}, attn {attn_torch.shape}")
    assert isinstance(output_torch, torch.Tensor)

    # Test 5: Different epsilon values
    for eps in [0.01, 0.1, 1.0]:
        attn_eps = HybridAttention(d_model=d_model, num_heads=num_heads, epsilon=eps)
        out_eps = attn_eps(x)
        has_nan = np.any(np.isnan(out_eps))
        print(f"  eps={eps}: output shape {out_eps.shape}, has_nan={has_nan}")
        assert not has_nan

    print("[HybridAttention] All tests passed!")
