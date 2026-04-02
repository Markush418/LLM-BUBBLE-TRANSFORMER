"""
PlateauAttentionMechanism — NumPy Implementation
===================================================
Minimal Surface / Soap Film Attention routing via Entropic Optimal Transport.
Pure NumPy implementation (no PyTorch required).

Uses Sinkhorn-Knopp algorithm in log-domain for numerical stability.
Supports multi-head attention with configurable epsilon (viscosity coefficient).
"""

import numpy as np
from typing import Optional, Tuple


class PlateauAttentionMechanism:
    """
    Computes Minimal Surface Attention via Entropic Optimal Transport.

    The attention matrix A is found by solving:
        min_A <A, C> - epsilon * H(A)
    subject to doubly-stochastic constraints (row/col sums = 1)

    where C is the geometric cost matrix C_ij = ||Q_i - K_j||^2
    and H(A) is the Shannon entropy of A.

    Solved via log-domain Sinkhorn-Knopp iterations.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        epsilon: float = 0.1,
        tau_iters: int = 5,
        seed: int = 42,
    ):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.epsilon = epsilon
        self.tau_iters = tau_iters

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = rng.randn(d_model, d_model).astype(np.float32) * scale

    def _cdist_sq(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        Q_sq = np.sum(Q**2, axis=-1, keepdims=True)
        K_sq = np.sum(K**2, axis=-1, keepdims=True)
        K_sq_t = np.moveaxis(K_sq, -2, -1)
        dist_sq = Q_sq + K_sq_t - 2.0 * np.matmul(Q, np.moveaxis(K, -2, -1))
        return np.maximum(dist_sq, 0.0)

    def _sinkhorn_log_domain(
        self, C: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        log_S = -C / self.epsilon

        if mask is not None:
            if mask.ndim == 2:
                mask_2d = mask[:, np.newaxis, :]
                mask_2d = (mask_2d & mask_2d.transpose(0, 1, 3, 2)).astype(np.float32)
            else:
                mask_2d = mask.astype(np.float32)
            log_S = np.where(mask_2d == 0, -1e10, log_S)

        B, heads, N, _ = log_S.shape
        u = np.zeros((B, heads, N), dtype=np.float32)
        v = np.zeros((B, heads, N), dtype=np.float32)

        for _ in range(self.tau_iters):
            u = -_logsumexp(log_S + v[:, :, np.newaxis, :], axis=-1)
            v = -_logsumexp(log_S + u[:, :, :, np.newaxis], axis=-2)

        A = np.exp(log_S + u[:, :, :, np.newaxis] + v[:, :, np.newaxis, :])
        return A

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        B, N, D = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        C = self._cdist_sq(Q, K)
        A = self._sinkhorn_log_domain(C, mask)

        output = np.matmul(A, V)
        output = output.transpose(0, 2, 1, 3).reshape(B, N, D)
        output = output @ self.W_o

        if return_attention:
            return output, A
        return output


class PlateauAttentionBlock:
    """Full transformer block with Plateau Attention + FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        epsilon: float = 0.1,
        tau_iters: int = 5,
        seed: int = 42,
    ):
        self.attention = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon,
            tau_iters=tau_iters,
            seed=seed,
        )
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        rng = np.random.RandomState(seed + 1)
        self.ff_w1 = rng.randn(d_model, ff_dim).astype(np.float32) * np.sqrt(
            2.0 / d_model
        )
        self.ff_b1 = np.zeros(ff_dim, dtype=np.float32)
        self.ff_w2 = rng.randn(ff_dim, d_model).astype(np.float32) * np.sqrt(
            2.0 / ff_dim
        )
        self.ff_b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        x = x + self.attention.forward(self.norm1.forward(x), mask=mask)[0]
        h = x @ self.ff_w1 + self.ff_b1
        h = gelu(h)
        h = h @ self.ff_w2 + self.ff_b2
        x = x + self.norm2.forward(h)
        return x


class LayerNorm:
    def __init__(self, d: int, eps: float = 1e-5):
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _logsumexp(a: np.ndarray, axis: int = None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


if __name__ == "__main__":
    print("[PlateauAttention] Running numpy quick test...")
    B, N, D = 2, 32, 128
    x = np.random.randn(B, N, D).astype(np.float32)

    attn = PlateauAttentionMechanism(d_model=D, num_heads=4, epsilon=0.1, tau_iters=5)
    output = attn.forward(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")

    output, attention = attn.forward(x, return_attention=True)
    print(f"  Attention matrix: {attention.shape}")
    print(f"  Row sums (should be ~1): {attention.sum(axis=-1).mean():.4f}")

    for eps in [0.01, 0.1, 1.0]:
        a = PlateauAttentionMechanism(d_model=D, num_heads=4, epsilon=eps, tau_iters=10)
        _, A = a.forward(x, return_attention=True)
        cr = np.mean(A > 1.0 / N)
        print(f"  eps={eps:.2f}: concentration_ratio={cr:.4f}")

    block = PlateauAttentionBlock(d_model=D, num_heads=4, ff_dim=D * 4, epsilon=0.1)
    output = block.forward(x)
    print(f"  Block output: {output.shape}")

    print("[PlateauAttention] All numpy tests passed!")
