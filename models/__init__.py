"""
Bubble Transformer V3 — legacy modules (SDOT replaced by HybridAttention)
==========================================================================

History (April-June 2026):
  - Original implementation: SDOT (Semi-Discrete Optimal Transport) with Voronoi
  - Migration (June 2026): SDOT replaced by HybridAttention (DeltaNet + SIRI + psi)
  - Current: This package retains core utilities (BaroreceptorMLP, v3/v4 helpers)
    that are still used by other modules. SDOT-specific code moved to
    docs/legacy/sdot_v3_v4/.

Modules:
  - v3_core: Core utilities (clustering, assignment, block attention)
  - baroreceptor: Dynamic C prediction MLP
  - v4_core: V4 routing utilities (FPS, expert-choice)
  - optimizers: Riemannian optimizers
"""

from .v3_core import cluster_keys, voronoi_assign, block_masked_attention
from .baroreceptor import BaroreceptorMLP

__all__ = [
    "cluster_keys",
    "voronoi_assign",
    "block_masked_attention",
    "BaroreceptorMLP",
    # SDOT-related classes (SDOTAttention, DualHeadSDOTAttentionV4) are deprecated.
    # See docs/legacy/sdot_v3_v4/ for the historical code.
]
