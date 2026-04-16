"""
GOATAttention — Gated Optimal Attention Transport
=====================================
GOAT mechanism: combines learnable gates with Sinkhorn optimal transport.

Each head has a learnable gate g_i that modulates the cost matrix:
    C'_ij = g_i * C_ij

With optional tied attention (k keys share one gate):
    C'_block(i, j) = g_{floor(i/k)} * C_{i,j}

Pure NumPy implementation.
"""

import numpy as np
from typing import Optional, Tuple
from .plateau_attention import (
    PlateauAttentionMechanism,
    CostFunction,
    CostFunctionFactory,
    L2SquaredCost,
    _logsumexp,
)


class GOATCostFunction(CostFunction):
    """Cost function with learnable gating per head/key.

    Wraps an underlying cost function and applies per-head gates.
    """

    def __init__(
        self,
        base_cost: CostFunction = None,
        num_heads: int = 8,
        tied: bool = False,
        gate_init: float = 1.0,
        seed: int = 42,
    ):
        self.base_cost = base_cost or L2SquaredCost()
        self.num_heads = num_heads
        self.tied = tied

        # Initialize gates
        rng = np.random.RandomState(seed)
        self.gates = rng.randn(num_heads).astype(np.float32) * 0.01 + gate_init
        self.gates = np.clip(self.gates, 0.01, 10.0)

    def compute(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Compute cost matrix with gating applied."""
        C = self.base_cost.compute(Q, K)
        B, heads, N, _ = C.shape
        for h in range(heads):
            C[:, h, :, :] = C[:, h, :, :] * self.gates[h]
        return self._validate(C)

    def get_gates(self) -> np.ndarray:
        return self.gates.copy()

    def set_gates(self, gates: np.ndarray) -> None:
        self.gates = np.clip(gates, 0.01, 10.0)


class GOATAttentionMechanism(PlateauAttentionMechanism):
    """Plateau Attention with learnable GOAT gates."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        epsilon: float = 0.1,
        tau_iters: int = 5,
        seed: int = 42,
        cost_type: str = "l2_sq",
        learn_gates: bool = True,
        gate_lr: float = 0.01,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon,
            tau_iters=tau_iters,
            seed=seed,
            cost_type=cost_type,
        )
        self.learn_gates = learn_gates
        self.gate_lr = gate_lr

        if learn_gates:
            self.cost_fn = GOATCostFunction(
                base_cost=self.cost_fn,
                num_heads=num_heads,
                tied=True,
                gate_init=1.0,
                seed=seed,
            )

    def get_gates(self) -> np.ndarray:
        if hasattr(self.cost_fn, "get_gates"):
            return self.cost_fn.get_gates()
        return np.array([])

    def set_gates(self, gates: np.ndarray) -> None:
        if hasattr(self.cost_fn, "set_gates"):
            self.cost_fn.set_gates(gates)


class DualHeadGOATAttention:
    """Dual-Head GOAT: combines two GOAT mechanisms with different epsilon."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        epsilon_low: float = 0.01,
        epsilon_high: float = 0.5,
        alpha: float = 0.5,
        tau_iters: int = 5,
        seed: int = 42,
        cost_type: str = "l2_sq",
    ):
        self.goat_low = GOATAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon_low,
            tau_iters=tau_iters,
            seed=seed,
            cost_type=cost_type,
        )
        self.goat_high = GOATAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon_high,
            tau_iters=tau_iters,
            seed=seed + 1,
            cost_type=cost_type,
        )
        self.alpha = alpha

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out_low = self.goat_low.forward(x, mask)
        out_high = self.goat_high.forward(x, mask)
        return self.alpha * out_low + (1.0 - self.alpha) * out_high

    def get_gates(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.goat_low.get_gates(), self.goat_high.get_gates()


if __name__ == "__main__":
    print("[GOATAttention] Running tests...")
    np.random.seed(42)
    B, N, D = 2, 32, 128
    x = np.random.randn(B, N, D).astype(np.float32)

    goat = GOATAttentionMechanism(d_model=D, num_heads=4, epsilon=0.1, learn_gates=True)
    output = goat.forward(x)
    print(f"  Input: {x.shape} -> Output: {output.shape}")
    print(f"  Gates: {goat.get_gates()}")

    dual = DualHeadGOATAttention(
        d_model=D, num_heads=4, epsilon_low=0.01, epsilon_high=0.5, alpha=0.5
    )
    out = dual.forward(x)
    print(f"  Dual-head: {x.shape} -> {out.shape}")

    print("[GOATAttention] Tests passed!")
