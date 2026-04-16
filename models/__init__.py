"""
Bubble Transformer V3 — SDOT Attention Modules
==============================================

Semi-Discrete Optimal Transport (SDOT) implementation.
Replaces Sinkhorn iterations with Voronoi assignment.

Modules:
- v3_core: Core SDOT algorithms (clustering, assignment, block attention)
- baroreceptor: Dynamic C prediction MLP
- sdot_attention: Complete SDOTAttention module
"""

from .v3_core import cluster_keys, voronoi_assign, block_masked_attention
from .baroreceptor import BaroreceptorMLP
from .sdot_attention import SDOTAttention

__all__ = [
    "cluster_keys",
    "voronoi_assign",
    "block_masked_attention",
    "BaroreceptorMLP",
    "SDOTAttention",
]
