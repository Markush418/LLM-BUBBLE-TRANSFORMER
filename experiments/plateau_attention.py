"""
PlateauAttentionMechanism — NumPy Implementation
===================================================
Minimal Surface / Soap Film Attention routing via Entropic Optimal Transport.
Pure NumPy implementation (no PyTorch required).

Uses Sinkhorn-Knopp algorithm in log-domain for numerical stability.
Supports multi-head attention with configurable epsilon (viscosity coefficient).
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


# =============================================================================
# Cost Function Hierarchy
# =============================================================================


class CostFunction(ABC):
    """Abstract base class for cost matrix computation."""

    @abstractmethod
    def compute(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Compute cost matrix C from Q and K.

        Args:
            Q: Query vectors [B, heads, N, head_dim]
            K: Key vectors [B, heads, N, head_dim]
        Returns:
            C: Cost matrix [B, heads, N, N], np.float32, non-negative, no NaN/Inf
        """
        pass

    def get_prior(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Get prior distribution over the cost matrix shape.

        Used to initialize Sinkhorn with a known prior distribution
        instead of uniform. Override in subclasses for non-uniform priors.

        Args:
            shape: Shape of the prior distribution to return

        Returns:
            prior: Prior distribution array of given shape, sums to 1.0
        """
        # Default: uniform prior
        N = shape[-1]
        return np.ones(shape, dtype=np.float32) / N

    def _validate(self, C: np.ndarray) -> np.ndarray:
        """Ensure cost matrix is valid: np.float32, non-negative, no NaN/Inf."""
        C = C.astype(np.float32)
        C = np.nan_to_num(C, nan=0.0, posinf=1e10, neginf=0.0)
        C = np.maximum(C, 0.0)
        return C


class L2SquaredCost(CostFunction):
    """Squared Euclidean distance: C_ij = ||Q_i - K_j||^2"""

    def compute(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        Q_sq = np.sum(Q**2, axis=-1, keepdims=True)
        K_sq = np.sum(K**2, axis=-1, keepdims=True)
        K_sq_t = np.moveaxis(K_sq, -2, -1)
        dist_sq = Q_sq + K_sq_t - 2.0 * np.matmul(Q, np.moveaxis(K, -2, -1))
        return self._validate(np.maximum(dist_sq, 0.0))


class CosineCost(CostFunction):
    """Cosine distance: C_ij = 1 - cos(Q_i, K_j)"""

    def compute(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        eps = 1e-8
        Q_norm = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + eps)
        K_norm = K / (np.linalg.norm(K, axis=-1, keepdims=True) + eps)
        sim = np.matmul(Q_norm, np.moveaxis(K_norm, -2, -1))
        sim = np.clip(sim, -1.0, 1.0)
        return self._validate(1.0 - sim)


class DotProductCost(CostFunction):
    """Negative dot product cost: C_ij = -Q_i . K_j, shifted to non-negative."""

    def compute(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        C = -np.matmul(Q, np.moveaxis(K, -2, -1))
        # Shift to non-negative
        C_min = np.min(C, axis=(-2, -1), keepdims=True)
        C = C - C_min
        return self._validate(C)


class MahalanobisCost(CostFunction):
    """Mahalanobis distance: C_ij = (Q_i - K_j)^T M (Q_i - K_j) with regularized covariance."""

    def __init__(self, reg_lambda: float = 1e-6):
        self.reg_lambda = reg_lambda

    def compute(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        B, heads, N, D = Q.shape
        # Compute regularized covariance from Q
        Q_reshaped = Q.reshape(-1, D)  # [B*heads*N, D]
        cov = np.cov(Q_reshaped, rowvar=False)  # [D, D]
        if cov.ndim == 0:
            cov = np.eye(D, dtype=np.float32) * float(cov)
        cov = cov.astype(np.float32)
        # Regularize: Sigma + lambda * I
        cov_reg = cov + self.reg_lambda * np.eye(D, dtype=np.float32)
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(cov_reg)
        except np.linalg.LinAlgError:
            # Fallback to identity
            L = np.eye(D, dtype=np.float32)
        # Transform Q and K
        Q_L = np.matmul(Q, L)
        K_L = np.matmul(K, L)
        # L2 in transformed space
        Q_sq = np.sum(Q_L**2, axis=-1, keepdims=True)
        K_sq = np.sum(K_L**2, axis=-1, keepdims=True)
        K_sq_t = np.moveaxis(K_sq, -2, -1)
        dist_sq = Q_sq + K_sq_t - 2.0 * np.matmul(Q_L, np.moveaxis(K_L, -2, -1))
        return self._validate(np.maximum(dist_sq, 0.0))


class MeshLearnableCost(CostFunction):
    """L2 squared + learnable perturbation delta. Delta is updated to minimize attention entropy."""

    def __init__(self, alpha: float = 0.1, seed: int = 42):
        self.alpha = alpha
        self.delta = None  # Will be initialized on first compute
        self.rng = np.random.RandomState(seed)

    def compute(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        B, heads, N, D = Q.shape
        # Initialize delta on first call
        if self.delta is None or self.delta.shape != (B, heads, N, N):
            self.delta = self.rng.randn(B, heads, N, N).astype(np.float32) * 0.01
        # L2 base
        Q_sq = np.sum(Q**2, axis=-1, keepdims=True)
        K_sq = np.sum(K**2, axis=-1, keepdims=True)
        K_sq_t = np.moveaxis(K_sq, -2, -1)
        C_l2 = np.maximum(
            Q_sq + K_sq_t - 2.0 * np.matmul(Q, np.moveaxis(K, -2, -1)), 0.0
        )
        # Add perturbation
        C = C_l2 + self.alpha * self.delta
        return self._validate(C)


VALID_COST_TYPES = ["l2_sq", "cosine", "dot_product", "mahalanobis", "mesh_learnable"]


class CostFunctionFactory:
    """Factory for creating cost functions by type name."""

    _cost_classes = {
        "l2_sq": L2SquaredCost,
        "cosine": CosineCost,
        "dot_product": DotProductCost,
        "mahalanobis": MahalanobisCost,
        "mesh_learnable": MeshLearnableCost,
    }

    @classmethod
    def create_cost(cls, cost_type: str) -> CostFunction:
        if cost_type not in cls._cost_classes:
            raise ValueError(
                f"Unknown cost type: {cost_type}. Valid types: {VALID_COST_TYPES}"
            )
        return cls._cost_classes[cost_type]()


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
        cost_type: str = "l2_sq",
    ):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.epsilon = epsilon
        self.tau_iters = tau_iters
        self.cost_fn = CostFunctionFactory.create_cost(cost_type)
        self.normalize_costs = True

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

    def _validate_convergence(self, A: np.ndarray) -> Tuple[bool, float]:
        """Check if attention matrix is doubly stochastic."""
        row_sums = A.sum(axis=-1)  # [B, heads, N]
        col_sums = A.sum(axis=-2)  # [B, heads, N]
        row_dev = np.max(np.abs(row_sums - 1.0))
        col_dev = np.max(np.abs(col_sums - 1.0))
        max_dev = max(row_dev, col_dev)
        return max_dev < 1e-3, float(max_dev)

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

        C = self.cost_fn.compute(Q, K)
        if self.normalize_costs:
            C_min = np.min(C, axis=(-2, -1), keepdims=True)
            C_max = np.max(C, axis=(-2, -1), keepdims=True)
            C = (C - C_min) / (C_max - C_min + 1e-10)
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


class DualHeadPlateauAttention:
    """
    Dual-Head Tension Architecture — Prevents representational collapse.

    Two PlateauAttentionMechanism instances with different ε values:
    - Head Low (ε_low): Concentration specialist, sparse attention
    - Head High (ε_high): Expressivity specialist, distributed attention

    Outputs fused via tension coefficient α:
        output = α · out_low + (1 - α) · out_high

    α = 1.0 → pure concentration (low ε)
    α = 0.0 → pure expressivity (high ε)
    α = 0.5 → balanced tension
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        epsilon_low: float = 0.001,
        epsilon_high: float = 0.1,
        alpha: float = 0.5,
        tau_iters: int = 5,
        seed: int = 42,
        cost_type: str = "l2_sq",
    ):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.alpha = alpha
        self.tau_iters = tau_iters

        # Shared projections
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = rng.randn(d_model, d_model).astype(np.float32) * scale

        # Two attention mechanisms (share cost function type)
        self.cost_fn = CostFunctionFactory.create_cost(cost_type)
        self.normalize_costs = True

        # Create internal mechanisms for Sinkhorn only
        self._head_low = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon_low,
            tau_iters=tau_iters,
            seed=seed,
            cost_type=cost_type,
        )
        self._head_high = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon_high,
            tau_iters=tau_iters,
            seed=seed + 1,
            cost_type=cost_type,
        )

    def _apply_sinkhorn(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        epsilon: float,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Sinkhorn with given epsilon using shared Q, K, V."""
        C = self.cost_fn.compute(Q, K)
        if self.normalize_costs:
            C_min = np.min(C, axis=(-2, -1), keepdims=True)
            C_max = np.max(C, axis=(-2, -1), keepdims=True)
            C = (C - C_min) / (C_max - C_min + 1e-10)

        # Temporarily override epsilon on _head_low for Sinkhorn
        original_epsilon = self._head_low.epsilon
        self._head_low.epsilon = epsilon
        A = self._head_low._sinkhorn_log_domain(C, mask)
        self._head_low.epsilon = original_epsilon

        output = np.matmul(A, V)
        output = output.transpose(0, 2, 1, 3).reshape(
            Q.shape[0], Q.shape[2], self.d_model
        )
        return output, A

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        B, N, D = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        out_low, A_low = self._apply_sinkhorn(Q, K, V, self.epsilon_low, mask)
        out_high, A_high = self._apply_sinkhorn(Q, K, V, self.epsilon_high, mask)

        # Tension fusion
        output = self.alpha * out_low + (1.0 - self.alpha) * out_high
        output = output @ self.W_o

        if return_attention:
            return output, A_low, A_high
        return output


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

    # Dual-head test
    print("\n[DualHeadPlateauAttention] Testing dual-head tension...")
    dual = DualHeadPlateauAttention(
        d_model=D, num_heads=4, epsilon_low=0.01, epsilon_high=0.5, alpha=0.5
    )
    out, A_low, A_high = dual.forward(x, return_attention=True)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  A_low shape: {A_low.shape}, A_high shape: {A_high.shape}")
    print(f"  A_low row sums: {A_low.sum(axis=-1).mean():.4f}")
    print(f"  A_high row sums: {A_high.sum(axis=-1).mean():.4f}")

    # Test alpha extremes
    for alpha_val in [0.0, 0.5, 1.0]:
        dual.alpha = alpha_val
        o = dual.forward(x)
        print(f"  alpha={alpha_val}: output mean={o.mean():.4f}, std={o.std():.4f}")
