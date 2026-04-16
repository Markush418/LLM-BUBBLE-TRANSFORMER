"""
SDOT Attention Module — Bubble Transformer V3
=============================================

Complete Semi-Discrete Optimal Transport Attention module.
Drop-in replacement for PlateauAttentionMechanism (V2).

Key differences from V2:
- No Sinkhorn iterations (replaced by Voronoi assignment)
- Complexity: O(N log C) vs O(N² × τ)
- Hard sparsity (bubbles) vs soft sparsity (Sinkhorn)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from .v3_core import cluster_keys, voronoi_assign, block_masked_attention
    from .baroreceptor import BaroreceptorMLP
except ImportError:
    # Standalone execution
    from v3_core import cluster_keys, voronoi_assign, block_masked_attention
    from baroreceptor import BaroreceptorMLP


class SDOTAttention(nn.Module):
    """
    Bubble Transformer V3: Semi-Discrete Optimal Transport Attention.

    Replaces Sinkhorn iterativo por:
    1. Clustering de Keys en C centroides
    2. Asignación Voronoi de Queries
    3. Atención en bloques enmascarados

    Complejidad: O(N log C) vs O(N² × τ) de Sinkhorn

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_centroids: Number of centroids C (fixed, or None for dynamic)
        use_baroreceptor: If True, use BaroreceptorMLP to predict C dynamically
        min_C: Minimum C (only used if use_baroreceptor=True)
        max_C: Maximum C (only used if use_baroreceptor=True)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_centroids: int = 32,
        use_baroreceptor: bool = True,
        min_C: int = 16,
        max_C: int = 512,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_centroids = num_centroids
        self.use_baroreceptor = use_baroreceptor

        # Standard projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Baroreceptor for dynamic C
        if use_baroreceptor:
            self.baroreceptor = BaroreceptorMLP(d_model, min_C, max_C)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_assignments: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with SDOT.

        Args:
            x: [B, N, d_model]
            attention_mask: [B, N] (optional, not used in current implementation)
            return_assignments: If True, return bubble assignments

        Returns:
            output: [B, N, d_model]
            assignments: [B, heads, N] (if return_assignments=True)
        """
        B, N, D = x.shape

        # 1. Projections
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Determine C
        if self.use_baroreceptor:
            C = self.baroreceptor(x)
        else:
            C = self.num_centroids

        # 3. Key clustering
        centroids = cluster_keys(K, num_centroids=C)

        # 4. Voronoi assignment
        assignments = voronoi_assign(Q, centroids)

        # 5. Block-masked attention
        output = block_masked_attention(Q, K, V, assignments, centroids)

        # 6. Final projection
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.W_o(output)

        if return_assignments:
            return output, assignments
        return output, None

    def forward_with_fixed_C(
        self,
        x: torch.Tensor,
        C: int,
        attention_mask: Optional[torch.Tensor] = None,
        return_assignments: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with fixed C (for calibration).

        Args:
            x: [B, N, d_model]
            C: Number of centroids (fixed)
            attention_mask: [B, N] (optional)
            return_assignments: If True, return bubble assignments

        Returns:
            output: [B, N, d_model]
            assignments: [B, heads, N] (if return_assignments=True)
        """
        B, N, D = x.shape

        # Projections
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Key clustering with fixed C
        centroids = cluster_keys(K, num_centroids=C)

        # Voronoi assignment
        assignments = voronoi_assign(Q, centroids)

        # Block-masked attention
        output = block_masked_attention(Q, K, V, assignments, centroids)

        # Final projection
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.W_o(output)

        if return_assignments:
            return output, assignments
        return output, None


if __name__ == "__main__":
    # Quick test
    print("[sdot_attention] Running quick test...")

    B, N, d_model = 2, 128, 512
    num_heads = 8

    x = torch.randn(B, N, d_model)

    # Test with fixed C
    sdot_fixed = SDOTAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_centroids=32,
        use_baroreceptor=False,
    )

    output, assignments = sdot_fixed(x, return_assignments=True)
    print(f"Fixed C: input {x.shape} -> output {output.shape}")
    assert output.shape == (B, N, d_model), (
        f"Expected {(B, N, d_model)}, got {output.shape}"
    )
    assert assignments is not None, "Assignments should not be None"
    print(f"Assignments: {assignments.shape}")

    # Test with dynamic C (baroreceptor)
    sdot_dynamic = SDOTAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_baroreceptor=True,
        min_C=16,
        max_C=128,
    )

    output_dyn, assignments_dyn = sdot_dynamic(x, return_assignments=True)
    print(f"Dynamic C: input {x.shape} -> output {output_dyn.shape}")
    assert output_dyn.shape == (B, N, d_model), (
        f"Expected {(B, N, d_model)}, got {output_dyn.shape}"
    )

    # Test forward_with_fixed_C
    output_fixed, assignments_fixed = sdot_dynamic.forward_with_fixed_C(
        x, C=64, return_assignments=True
    )
    print(f"forward_with_fixed_C(C=64): output {output_fixed.shape}")
    assert output_fixed.shape == (B, N, d_model), (
        f"Expected {(B, N, d_model)}, got {output_fixed.shape}"
    )

    print("[sdot_attention] All tests passed!")
