"""
SDOT Attention V4 — Bubble Transformer V4
=========================================

Complete Semi-Discrete Optimal Transport Attention module for V4.
Drop-in replacement for V3's SDOTAttention.

Key innovations over V3:
1. FPS initialization for better centroid coverage
2. Expert-Choice routing for balanced load
3. Power Diagrams (Laguerre cells) for variable bubble sizes
4. Manifold-aware centroids (Euclidean, Poincaré, Stiefel, Sphere)
5. Warm-start from previous layer centroids

Complexity: O(N * C) for FPS, O(N * k) for routing
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

try:
    from .v4_core import (
        fps_initialize_centroids,
        expert_choice_routing,
        routed_attention,
        power_diagram_assign,
        warm_start_centroids,
        compute_routing_balance,
        compute_coverage,
    )
    from .baroreceptor import BaroreceptorMLP
    from .bubble_centroids_v4 import BubbleCentroidsV4, ManifoldType
except ImportError:
    # Standalone execution
    from v4_core import (
        fps_initialize_centroids,
        expert_choice_routing,
        routed_attention,
        power_diagram_assign,
        warm_start_centroids,
        compute_routing_balance,
        compute_coverage,
    )
    from baroreceptor import BaroreceptorMLP
    from bubble_centroids_v4 import BubbleCentroidsV4, ManifoldType


class SDOTAttentionV4(nn.Module):
    """
    Bubble Transformer V4: Semi-Discrete Optimal Transport Attention.

    Integrates:
    1. FPS initialization for centroids (better coverage than random)
    2. Expert-Choice routing (balanced load per expert)
    3. Power Diagrams (optional, for variable bubble sizes)
    4. Manifold-aware centroids (Euclidean, Poincaré, Stiefel, Sphere)
    5. Warm-start from previous layer

    This is a drop-in replacement for V3's SDOTAttention.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads (Q heads)
        num_kv_heads: Number of KV heads for GQA (default: None = num_heads)
            - When num_kv_heads < num_heads, uses Grouped Query Attention
            - Each KV head is shared by (num_heads // num_kv_heads) Q heads
            - Example: num_heads=16, num_kv_heads=8 -> each KV head serves 2 Q heads
            - Reduces KV cache memory by (1 - num_kv_heads/num_heads)
        num_centroids: Number of centroids C (default: 32)
        use_baroreceptor: Use BaroreceptorMLP to predict C dynamically (default: True)
        use_fps_init: Use FPS initialization (default: True)
        use_power_diagrams: Enable Power Diagrams mode (default: False)
        use_expert_routing: Enable Expert-Choice routing (default: True)
        manifold_type: Manifold type for centroids (default: 'euclidean')
            - 'euclidean': Standard R^d space
            - 'poincare': Hyperbolic space for hierarchical data
            - 'stiefel': Orthonormal frames
            - 'sphere': Unit sphere for normalized embeddings
        min_C: Minimum C (only used if use_baroreceptor=True)
        max_C: Maximum C (only used if use_baroreceptor=True)
        top_k: Tokens per expert for routing (default: 8)
        temperature: Routing temperature (default: 1.0)
        warm_start_alpha: Blending coefficient for warm-start (default: 0.7)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,  # NEW: GQA support
        num_centroids: int = 32,
        use_baroreceptor: bool = True,
        use_fps_init: bool = True,
        use_power_diagrams: bool = False,
        use_expert_routing: bool = True,
        manifold_type: str = "euclidean",
        min_C: int = 16,
        max_C: int = 512,
        top_k: int = 8,
        temperature: float = 1.0,
        warm_start_alpha: float = 0.7,
    ):
        super().__init__()

        # Validate d_model divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # GQA support: num_kv_heads defaults to num_heads (MHA)
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_repetition = num_heads // self.num_kv_heads

        # Validate GQA configuration
        assert num_heads % self.num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_centroids = num_centroids
        self.use_baroreceptor = use_baroreceptor
        self.use_fps_init = use_fps_init
        self.use_power_diagrams = use_power_diagrams
        self.use_expert_routing = use_expert_routing
        self.manifold_type = manifold_type
        self.min_C = min_C
        self.max_C = max_C
        self.top_k = top_k
        self.temperature = temperature
        self.warm_start_alpha = warm_start_alpha

        # Projections with GQA support
        # W_q: Full Q heads [d_model, num_heads * head_dim]
        # W_k, W_v: Reduced KV heads [d_model, num_kv_heads * head_dim]
        # W_o: Back to d_model [num_heads * head_dim, d_model]
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(num_heads * self.head_dim, d_model)

        # Baroreceptor for dynamic C (same as V3)
        if use_baroreceptor:
            self.baroreceptor = BaroreceptorMLP(d_model, min_C, max_C)

        # Learnable centroids (optional, for non-FPS mode)
        # Only used if use_fps_init=False
        if not use_fps_init:
            self.bubble_centroids = BubbleCentroidsV4(
                num_heads=num_heads,
                num_experts=num_centroids,
                head_dim=self.head_dim,
                manifold_type=manifold_type,
                learnable=True,
            )

        # Power Diagram weights ψ (one per head per centroid)
        # Shape: [1, num_heads, num_centroids]
        # Zero initialization: ψ=0 reduces to standard Voronoi
        if use_power_diagrams:
            self.psi = nn.Parameter(torch.zeros(1, num_heads, num_centroids))
        else:
            self.psi = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_assignments: bool = False,
        previous_centroids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass with SDOT V4.

        Args:
            x: [B, N, d_model] - input tokens
            attention_mask: [B, N] (optional, not used in current implementation)
            return_assignments: If True, return assignment info
            previous_centroids: [B, H, C, d_head] - warm-start from previous layer

        Returns:
            output: [B, N, d_model]
            assignments_info: dict with assignments and routing stats (if return_assignments=True)
        """
        B, N, D = x.shape

        # 1. Projections with GQA support
        # Q: [B, N, num_heads * head_dim] -> [B, num_heads, N, head_dim]
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: [B, N, num_kv_heads * head_dim] -> [B, num_kv_heads, N, head_dim]
        K = self.W_k(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K, V for GQA (if needed)
        # When num_kv_heads < num_heads, repeat each KV head to match Q heads
        # Example: 8 KV heads -> 16 Q heads (repetition=2)
        if self.head_repetition > 1:
            # [B, num_kv_heads, N, head_dim] -> [B, num_heads, N, head_dim]
            K = K.repeat_interleave(self.head_repetition, dim=1)
            V = V.repeat_interleave(self.head_repetition, dim=1)

        # 2. Determine C (number of centroids)
        if self.use_baroreceptor:
            C = self.baroreceptor(x)
        else:
            C = self.num_centroids

        # 3. Initialize centroids
        if self.use_fps_init:
            # FPS initialization from Keys
            centroids = fps_initialize_centroids(K, C)
        else:
            # Use learnable centroids
            centroids = self.bubble_centroids(B)
            # If C differs from num_centroids, adjust
            if C != self.num_centroids:
                # Interpolate or truncate centroids
                if C < self.num_centroids:
                    centroids = centroids[:, :, :C, :]
                else:
                    # Pad with repeated centroids (rare case)
                    pad_size = C - self.num_centroids
                    centroids = torch.cat(
                        [centroids, centroids[:, :, :pad_size, :]], dim=2
                    )

        # 4. Warm-start from previous layer (if provided)
        if previous_centroids is not None:
            centroids = warm_start_centroids(
                current=centroids,
                previous=previous_centroids,
                alpha=self.warm_start_alpha,
                manifold_type=self.manifold_type
                if self.manifold_type != "euclidean"
                else None,
            )

        # 5. Assignment phase
        assignments_info = {}

        if self.use_power_diagrams and self.psi is not None:
            # Power Diagram assignment (Laguerre cells)
            # ψ shape: [1, H, num_centroids] -> expand to [B, H, C]
            # Handle dynamic C from baroreceptor
            psi_C = self.psi.shape[2]
            if C != psi_C:
                # Adjust psi to match dynamic C
                if C < psi_C:
                    psi_adjusted = self.psi[:, :, :C]
                else:
                    # Pad with zeros (rare case)
                    pad = torch.zeros(
                        1, self.num_heads, C - psi_C, device=self.psi.device
                    )
                    psi_adjusted = torch.cat([self.psi, pad], dim=2)
                psi_expanded = psi_adjusted.expand(B, -1, -1)
            else:
                psi_expanded = self.psi.expand(B, -1, -1)

            # Use soft assignment for training (differentiable)
            # Use hard assignment for inference
            hard = not self.training

            assignments = power_diagram_assign(
                tokens=Q,
                centroids=centroids,
                psi=psi_expanded,
                hard=hard,
                temperature=self.temperature,
            )

            assignments_info["power_diagram_assignments"] = assignments
            assignments_info["psi"] = psi_expanded

        # 6. Expert-Choice routing (if enabled)
        if self.use_expert_routing:
            # Compute routing weights and expert mask
            routing_weights, expert_mask = expert_choice_routing(
                Q=Q,
                centroids=centroids,
                top_k=self.top_k,
                temperature=self.temperature,
            )

            assignments_info["routing_weights"] = routing_weights
            assignments_info["expert_mask"] = expert_mask

            # 7. Routed attention
            output = routed_attention(
                Q=Q,
                K=K,
                V=V,
                routing_weights=routing_weights,
                expert_mask=expert_mask,
                centroids=centroids,
            )
        else:
            # Fallback: standard attention without routing
            # (for backward compatibility or ablation studies)
            scale = 1.0 / (self.head_dim**0.5)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_weights, V)

        # 8. Final projection
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.W_o(output)

        # 9. Collect assignment info (if requested)
        if return_assignments:
            assignments_info["centroids"] = centroids

            # Compute routing statistics if expert routing is enabled
            if self.use_expert_routing:
                balance = compute_routing_balance(routing_weights)
                coverage = compute_coverage(expert_mask, N)
                assignments_info["balance"] = balance
                assignments_info["coverage"] = coverage

            return output, assignments_info

        return output, None

    def forward_with_fixed_C(
        self,
        x: torch.Tensor,
        C: int,
        attention_mask: Optional[torch.Tensor] = None,
        return_assignments: bool = False,
        previous_centroids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass with fixed C (for calibration).

        Args:
            x: [B, N, d_model]
            C: Number of centroids (fixed)
            attention_mask: [B, N] (optional)
            return_assignments: If True, return assignment info
            previous_centroids: [B, H, C, d_head] - warm-start from previous layer

        Returns:
            output: [B, N, d_model]
            assignments_info: dict (if return_assignments=True)
        """
        B, N, D = x.shape

        # Projections with GQA support
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K, V for GQA (if needed)
        if self.head_repetition > 1:
            K = K.repeat_interleave(self.head_repetition, dim=1)
            V = V.repeat_interleave(self.head_repetition, dim=1)

        # Initialize centroids with fixed C
        if self.use_fps_init:
            centroids = fps_initialize_centroids(K, C)
        else:
            centroids = self.bubble_centroids(B)
            if C != self.num_centroids:
                if C < self.num_centroids:
                    centroids = centroids[:, :, :C, :]
                else:
                    pad_size = C - self.num_centroids
                    centroids = torch.cat(
                        [centroids, centroids[:, :, :pad_size, :]], dim=2
                    )

        # Warm-start
        if previous_centroids is not None:
            centroids = warm_start_centroids(
                current=centroids,
                previous=previous_centroids,
                alpha=self.warm_start_alpha,
                manifold_type=self.manifold_type
                if self.manifold_type != "euclidean"
                else None,
            )

        # Assignment and routing
        assignments_info = {}

        if self.use_power_diagrams and self.psi is not None:
            psi_expanded = self.psi.expand(B, -1, -1)
            hard = not self.training
            assignments = power_diagram_assign(
                tokens=Q,
                centroids=centroids,
                psi=psi_expanded,
                hard=hard,
                temperature=self.temperature,
            )
            assignments_info["power_diagram_assignments"] = assignments

        if self.use_expert_routing:
            routing_weights, expert_mask = expert_choice_routing(
                Q=Q,
                centroids=centroids,
                top_k=self.top_k,
                temperature=self.temperature,
            )
            assignments_info["routing_weights"] = routing_weights
            assignments_info["expert_mask"] = expert_mask

            output = routed_attention(
                Q=Q,
                K=K,
                V=V,
                routing_weights=routing_weights,
                expert_mask=expert_mask,
                centroids=centroids,
            )
        else:
            scale = 1.0 / (self.head_dim**0.5)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_weights, V)

        # Final projection
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.W_o(output)

        if return_assignments:
            assignments_info["centroids"] = centroids
            if self.use_expert_routing:
                balance = compute_routing_balance(routing_weights)
                coverage = compute_coverage(expert_mask, N)
                assignments_info["balance"] = balance
                assignments_info["coverage"] = coverage
            return output, assignments_info

        return output, None


class DualHeadSDOTAttentionV4(nn.Module):
    """
    Dual-Head SDOT Attention V4 — Production dual-head tension architecture.

    Two :class:`SDOTAttentionV4` instances operate in parallel.  Their outputs
    are fused via a tension coefficient ``alpha``:

        output = alpha * out_low + (1 - alpha) * out_high

    When ``share_projections=True``, the Q/K/V/O projection *parameters* are
    shared between the two heads (the same :class:`torch.nn.Parameter` tensors
    back both heads, so optimisers see them exactly once).

    Both heads retain their own non-projection state (baroreceptor, centroids,
    Power-Diagram ``psi``, etc.), so they may produce different routing and
    assignment patterns even when projections are shared.

    Args:
        All arguments accepted by :class:`SDOTAttentionV4`.
        epsilon_low (float): Retained for API compatibility with the NumPy
            ``DualHeadPlateauAttention`` reference.  Default: ``0.001``.
        epsilon_high (float): Same as above.  Default: ``0.1``.
        alpha (float): Tension fusion coefficient.  ``1.0`` → pure low-head,
            ``0.0`` → pure high-head, ``0.5`` → balanced.  Default: ``0.5``.
        share_projections (bool): If ``True``, the ``W_q``, ``W_k``, ``W_v``
            and ``W_o`` parameters are shared between the two internal heads.
            Default: ``True``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        num_centroids: int = 32,
        use_baroreceptor: bool = True,
        use_fps_init: bool = True,
        use_power_diagrams: bool = False,
        use_expert_routing: bool = True,
        manifold_type: str = "euclidean",
        min_C: int = 16,
        max_C: int = 512,
        top_k: int = 8,
        temperature: float = 1.0,
        warm_start_alpha: float = 0.7,
        epsilon_low: float = 0.001,
        epsilon_high: float = 0.1,
        alpha: float = 0.5,
        share_projections: bool = True,
    ):
        super().__init__()

        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.alpha = alpha
        self.share_projections = share_projections

        # Build both internal heads with identical V4 configuration
        common_kwargs = {
            "d_model": d_model,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "num_centroids": num_centroids,
            "use_baroreceptor": use_baroreceptor,
            "use_fps_init": use_fps_init,
            "use_power_diagrams": use_power_diagrams,
            "use_expert_routing": use_expert_routing,
            "manifold_type": manifold_type,
            "min_C": min_C,
            "max_C": max_C,
            "top_k": top_k,
            "temperature": temperature,
            "warm_start_alpha": warm_start_alpha,
        }

        self.head_low = SDOTAttentionV4(**common_kwargs)
        self.head_high = SDOTAttentionV4(**common_kwargs)

        if share_projections:
            # Share the Parameter tensors of W_q/k/v/o between the two heads.
            # PyTorch's ``named_parameters()`` deduplicates by object identity,
            # so shared tensors appear exactly once in the optimiser state.
            for proj_name in ("W_q", "W_k", "W_v", "W_o"):
                low_proj = getattr(self.head_low, proj_name)
                high_proj = getattr(self.head_high, proj_name)
                high_proj.weight = low_proj.weight
                if low_proj.bias is not None and high_proj.bias is not None:
                    high_proj.bias = low_proj.bias

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_assignments: bool = False,
        previous_centroids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass with dual-head tension.

        Args:
            x: ``[B, N, d_model]`` — input tokens.
            attention_mask: ``[B, N]`` (optional).
            return_assignments: If ``True``, return per-head assignment info.
            previous_centroids: ``[B, H, C, d_head]`` — warm-start from
                previous layer (optional).

        Returns:
            output: ``[B, N, d_model]``
            assignments_info: Nested dict with keys ``head_low`` and
                ``head_high`` (if ``return_assignments=True``), else ``None``.
        """
        # Run both attention heads
        out_low, info_low = self.head_low(
            x,
            attention_mask=attention_mask,
            return_assignments=return_assignments,
            previous_centroids=previous_centroids,
        )
        out_high, info_high = self.head_high(
            x,
            attention_mask=attention_mask,
            return_assignments=return_assignments,
            previous_centroids=previous_centroids,
        )

        # Tension fusion
        output = self.alpha * out_low + (1.0 - self.alpha) * out_high

        if return_assignments:
            assignments_info: Dict[str, Any] = {
                "head_low": info_low,
                "head_high": info_high,
                "alpha": self.alpha,
            }
            return output, assignments_info

        return output, None


if __name__ == "__main__":
    # Quick test
    print("[sdot_attention_v4] Running quick test...")

    B, N, d_model = 2, 128, 512
    num_heads = 8

    x = torch.randn(B, N, d_model)

    # Test 1: Basic V4 with FPS + Expert-Choice (default)
    print("\n[Test 1] Basic V4 (FPS + Expert-Choice)")
    sdot_v4 = SDOTAttentionV4(
        d_model=d_model,
        num_heads=num_heads,
        num_centroids=32,
        use_baroreceptor=False,
        use_fps_init=True,
        use_expert_routing=True,
    )

    output, assignments = sdot_v4(x, return_assignments=True)
    print(f"  Input: {x.shape} -> Output: {output.shape}")
    assert output.shape == (B, N, d_model), (
        f"Expected {(B, N, d_model)}, got {output.shape}"
    )
    assert assignments is not None, "Assignments should not be None"
    print(f"  Centroids shape: {assignments['centroids'].shape}")
    print(f"  Balance: {assignments['balance'].mean().item():.4f}")
    print(f"  Coverage: {assignments['coverage'].mean().item():.4f}")

    # Test 2: V4 with Power Diagrams
    print("\n[Test 2] V4 with Power Diagrams")
    sdot_v4_pd = SDOTAttentionV4(
        d_model=d_model,
        num_heads=num_heads,
        num_centroids=32,
        use_baroreceptor=False,
        use_fps_init=True,
        use_power_diagrams=True,
        use_expert_routing=True,
    )

    output_pd, assignments_pd = sdot_v4_pd(x, return_assignments=True)
    print(f"  Input: {x.shape} -> Output: {output_pd.shape}")
    assert output_pd.shape == (B, N, d_model)
    assert "power_diagram_assignments" in assignments_pd, (
        "Power Diagram assignments missing"
    )
    print(
        f"  Power Diagram assignments: {assignments_pd['power_diagram_assignments'].shape}"
    )
    print(f" psi shape: {assignments_pd['psi'].shape}")

    # Test 3: V4 with Baroreceptor (dynamic C)
    print("\n[Test 3] V4 with Baroreceptor (dynamic C)")
    sdot_v4_dynamic = SDOTAttentionV4(
        d_model=d_model,
        num_heads=num_heads,
        use_baroreceptor=True,
        min_C=16,
        max_C=128,
        use_fps_init=True,
    )

    output_dyn, assignments_dyn = sdot_v4_dynamic(x, return_assignments=True)
    print(f"  Input: {x.shape} -> Output: {output_dyn.shape}")
    assert output_dyn.shape == (B, N, d_model)

    # Test 4: V4 with warm-start from previous layer
    print("\n[Test 4] V4 with warm-start")
    previous_centroids = torch.randn(B, num_heads, 32, d_model // num_heads)

    output_ws, assignments_ws = sdot_v4(
        x,
        return_assignments=True,
        previous_centroids=previous_centroids,
    )
    print(f"  Input: {x.shape} -> Output: {output_ws.shape}")
    assert output_ws.shape == (B, N, d_model)

    # Test 5: V4 with fixed C (calibration mode)
    print("\n[Test 5] V4 with fixed C=64")
    output_fixed, assignments_fixed = sdot_v4_dynamic.forward_with_fixed_C(
        x, C=64, return_assignments=True
    )
    print(f"  Input: {x.shape} -> Output: {output_fixed.shape}")
    assert output_fixed.shape == (B, N, d_model)

    # Test 6: V4 without Expert-Choice routing (ablation)
    print("\n[Test 6] V4 without Expert-Choice routing")
    sdot_v4_no_routing = SDOTAttentionV4(
        d_model=d_model,
        num_heads=num_heads,
        num_centroids=32,
        use_baroreceptor=False,
        use_fps_init=True,
        use_expert_routing=False,  # Disable routing
    )

    output_no_routing, _ = sdot_v4_no_routing(x, return_assignments=False)
    print(f"  Input: {x.shape} -> Output: {output_no_routing.shape}")
    assert output_no_routing.shape == (B, N, d_model)

    # Test 7: V4 with learnable centroids (non-FPS mode)
    print("\n[Test 7] V4 with learnable centroids (non-FPS)")
    sdot_v4_learnable = SDOTAttentionV4(
        d_model=d_model,
        num_heads=num_heads,
        num_centroids=32,
        use_baroreceptor=False,
        use_fps_init=False,  # Use learnable centroids
        manifold_type="euclidean",
    )

    output_learn, assignments_learn = sdot_v4_learnable(x, return_assignments=True)
    print(f"  Input: {x.shape} -> Output: {output_learn.shape}")
    assert output_learn.shape == (B, N, d_model)

    print("\n[sdot_attention_v4] All tests passed!")
