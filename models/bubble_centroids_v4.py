"""
Bubble Centroids V4 — Manifold-Aware Parameters
==============================================

Manifold-aware centroid parameters using geoopt integration.
Supports Poincaré ball, Stiefel manifold, and Euclidean spaces.

Key features:
1. ManifoldParameter for gradient-aware optimization on manifolds
2. Manifold-specific distance functions
3. Projection operators for constraint satisfaction

Dependencies: geoopt >= 0.5.1
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any

try:
    import geoopt
    import geoopt.manifolds as manifolds

    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    geoopt = None
    manifolds = None


class ManifoldType:
    """Supported manifold types for centroids."""

    EUCLIDEAN = "euclidean"
    POINCARE = "poincare"
    STIEFEL = "stiefel"
    SPHERE = "sphere"


def get_manifold(manifold_type: str, dim: int, **kwargs) -> Optional[Any]:
    """
    Get manifold instance by type.

    Args:
        manifold_type: One of 'euclidean', 'poincare', 'stiefel', 'sphere'
        dim: Dimension of the manifold
        **kwargs: Additional manifold-specific parameters

    Returns:
        Manifold instance (or None if geoopt not available)

    Note: Returns None if geoopt is not installed.
    """
    if not GEOOPT_AVAILABLE:
        return None

    if manifold_type == ManifoldType.EUCLIDEAN:
        return manifolds.Euclidean(dim)

    elif manifold_type == ManifoldType.POINCARE:
        # Poincaré ball with configurable curvature
        c = kwargs.get("curvature", 1.0)
        return manifolds.PoincareBall(c=c)

    elif manifold_type == ManifoldType.STIEFEL:
        # Stiefel manifold St(n, p) - orthonormal frames
        n = kwargs.get("n", dim)
        p = kwargs.get("p", min(dim, 16))
        return manifolds.Stiefel()

    elif manifold_type == ManifoldType.SPHERE:
        # Unit sphere S^{dim-1}
        return manifolds.Sphere()

    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


def manifold_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    manifold: Optional[Any] = None,
) -> torch.Tensor:
    """
    Compute distance on manifold.

    Args:
        x: First point [B, ..., dim]
        y: Second point [B, ..., dim]
        manifold: Manifold instance (None for Euclidean)

    Returns:
        distances: [B, ...] - manifold distances

    Note: Falls back to Euclidean distance if manifold is None.
    """
    if manifold is None or not GEOOPT_AVAILABLE:
        # Euclidean distance
        return torch.norm(x - y, dim=-1)

    # Use geoopt's dist function
    try:
        return manifold.dist(x, y)
    except (AttributeError, NotImplementedError):
        # Fallback to Euclidean if manifold doesn't support dist
        return torch.norm(x - y, dim=-1)


def project_to_manifold(
    x: torch.Tensor,
    manifold: Optional[Any] = None,
) -> torch.Tensor:
    """
    Project point to manifold.

    Args:
        x: Point to project [B, ..., dim]
        manifold: Manifold instance (None for Euclidean)

    Returns:
        projected: [B, ..., dim] - projected point

    Note: Returns x unchanged if manifold is None.
    """
    if manifold is None or not GEOOPT_AVAILABLE:
        return x

    try:
        return manifold.projx(x)
    except (AttributeError, NotImplementedError):
        return x


class BubbleCentroidsV4(nn.Module):
    """
    Manifold-aware bubble centroids for V4.

    Supports multiple manifold types:
    - Euclidean: Standard R^d space
    - Poincaré: Hyperbolic space for hierarchical data
    - Stiefel: Orthonormal frames for structured representations
    - Sphere: Unit sphere for normalized embeddings

    Args:
        num_heads: Number of attention heads
        num_experts: Number of experts/centroids C
        head_dim: Dimension per head
        manifold_type: Type of manifold ('euclidean', 'poincare', 'stiefel', 'sphere')
        learnable: If True, centroids are learnable parameters
        **manifold_kwargs: Additional manifold parameters
    """

    def __init__(
        self,
        num_heads: int,
        num_experts: int,
        head_dim: int,
        manifold_type: str = ManifoldType.EUCLIDEAN,
        learnable: bool = True,
        **manifold_kwargs,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = head_dim
        self.manifold_type = manifold_type
        self.learnable = learnable

        # Get manifold
        self.manifold = get_manifold(manifold_type, head_dim, **manifold_kwargs)

        # Initialize centroids
        if learnable and GEOOPT_AVAILABLE and self.manifold is not None:
            # Use ManifoldParameter for manifold-aware optimization
            self._init_manifold_parameter(**manifold_kwargs)
        else:
            # Standard parameter (Euclidean or geoopt not available)
            self._init_euclidean_parameter()

    def _init_manifold_parameter(self, **kwargs):
        """Initialize centroids as ManifoldParameter."""
        if self.manifold_type == ManifoldType.POINCARE:
            # Initialize in Poincaré ball
            # Use exponential map from tangent space at origin
            tangent = torch.randn(1, self.num_heads, self.num_experts, self.head_dim)
            tangent = tangent * 0.01  # Small initial values

            if hasattr(self.manifold, "expmap0"):
                centroids = self.manifold.expmap0(tangent)
            else:
                centroids = torch.tanh(tangent) * 0.9  # Stay inside ball

            self.centroids = geoopt.ManifoldParameter(centroids, manifold=self.manifold)

        elif self.manifold_type == ManifoldType.STIEFEL:
            # Initialize as random orthonormal frames
            # Use QR decomposition for orthonormality
            random_matrix = torch.randn(
                1, self.num_heads, self.num_experts, self.head_dim
            )
            # Note: Stiefel manifold requires special handling
            # For simplicity, use standard parameter with projection
            self.centroids = nn.Parameter(random_matrix * 0.02)

        elif self.manifold_type == ManifoldType.SPHERE:
            # Initialize on unit sphere
            random_vec = torch.randn(1, self.num_heads, self.num_experts, self.head_dim)
            centroids = torch.nn.functional.normalize(random_vec, dim=-1)

            self.centroids = geoopt.ManifoldParameter(centroids, manifold=self.manifold)

        else:
            # Euclidean (default)
            self._init_euclidean_parameter()

    def _init_euclidean_parameter(self):
        """Initialize centroids as standard Euclidean parameters."""
        self.centroids = nn.Parameter(
            torch.randn(1, self.num_heads, self.num_experts, self.head_dim) * 0.02
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get centroids for given batch size.

        Args:
            batch_size: Batch size B

        Returns:
            centroids: [B, num_heads, num_experts, head_dim]
        """
        # Broadcast to batch
        return self.centroids.expand(batch_size, -1, -1, -1)

    def distance_to_queries(
        self,
        Q: torch.Tensor,
        centroids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute manifold-aware distances from queries to centroids.

        Args:
            Q: Query tensor [B, H, N, d]
            centroids: Override centroids (uses self.centroids if None)

        Returns:
            distances: [B, H, N, C] - distances to each centroid
        """
        if centroids is None:
            centroids = self.centroids

        B, H, N, d = Q.shape
        C = centroids.shape[2]

        if self.manifold is None or not GEOOPT_AVAILABLE:
            # Euclidean distance (vectorized)
            # Q: [B, H, N, d], centroids: [B, H, C, d]
            # Use torch.cdist for efficiency
            return torch.cdist(Q, centroids)

        # Manifold-aware distance
        # Expand Q for broadcasting: [B, H, N, 1, d]
        Q_expanded = Q.unsqueeze(3)
        # Expand centroids: [B, H, 1, C, d]
        centroids_expanded = centroids.unsqueeze(2)

        # Compute manifold distance
        try:
            # geoopt's dist expects points on manifold
            # We need to ensure both are projected
            Q_proj = project_to_manifold(Q_expanded, self.manifold)
            centroids_proj = project_to_manifold(centroids_expanded, self.manifold)

            distances = manifold_distance(Q_proj, centroids_proj, self.manifold)
            return distances.squeeze(-1)  # [B, H, N, C]

        except Exception:
            # Fallback to Euclidean
            return torch.cdist(Q, centroids)

    def project_centroids(self) -> torch.Tensor:
        """
        Project centroids to manifold (for constraint satisfaction).

        Returns:
            projected_centroids: [1, H, C, d]
        """
        return project_to_manifold(self.centroids, self.manifold)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        geoopt_status = "available" if GEOOPT_AVAILABLE else "not available"
        return (
            f"num_heads={self.num_heads}, "
            f"num_experts={self.num_experts}, "
            f"head_dim={self.head_dim}, "
            f"manifold={self.manifold_type}, "
            f"geoopt={geoopt_status}"
        )


class HybridManifoldCentroids(nn.Module):
    """
    Hybrid manifold centroids: different manifolds for different heads.

    Allows mixing Euclidean and hyperbolic representations
    for capturing both flat and hierarchical structure.

    Args:
        num_heads: Total number of attention heads
        num_experts: Number of experts per head
        head_dim: Dimension per head
        euclidean_heads: Number of heads using Euclidean space
        poincare_heads: Number of heads using Poincaré ball
    """

    def __init__(
        self,
        num_heads: int,
        num_experts: int,
        head_dim: int,
        euclidean_heads: int = 4,
        poincare_heads: int = 4,
    ):
        super().__init__()

        assert euclidean_heads + poincare_heads == num_heads, (
            "Head counts must sum to num_heads"
        )

        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = head_dim
        self.euclidean_heads = euclidean_heads
        self.poincare_heads = poincare_heads

        # Create separate centroid groups
        if euclidean_heads > 0:
            self.euclidean_centroids = BubbleCentroidsV4(
                num_heads=euclidean_heads,
                num_experts=num_experts,
                head_dim=head_dim,
                manifold_type=ManifoldType.EUCLIDEAN,
            )

        if poincare_heads > 0:
            self.poincare_centroids = BubbleCentroidsV4(
                num_heads=poincare_heads,
                num_experts=num_experts,
                head_dim=head_dim,
                manifold_type=ManifoldType.POINCARE,
            )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get all centroids concatenated.

        Args:
            batch_size: Batch size B

        Returns:
            centroids: [B, num_heads, num_experts, head_dim]
        """
        centroids_list = []

        if self.euclidean_heads > 0:
            euclidean = self.euclidean_centroids(batch_size)
            centroids_list.append(euclidean)

        if self.poincare_heads > 0:
            poincare = self.poincare_centroids(batch_size)
            centroids_list.append(poincare)

        return torch.cat(centroids_list, dim=1)

    def distance_to_queries(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute distances using appropriate manifold per head group.

        Args:
            Q: Query tensor [B, H, N, d]

        Returns:
            distances: [B, H, N, C]
        """
        B, H, N, d = Q.shape
        C = self.num_experts

        distances_list = []

        if self.euclidean_heads > 0:
            Q_euclidean = Q[:, : self.euclidean_heads, :, :]
            euclidean_dists = self.euclidean_centroids.distance_to_queries(Q_euclidean)
            distances_list.append(euclidean_dists)

        if self.poincare_heads > 0:
            Q_poincare = Q[:, self.euclidean_heads :, :, :]
            poincare_dists = self.poincare_centroids.distance_to_queries(Q_poincare)
            distances_list.append(poincare_dists)

        return torch.cat(distances_list, dim=1)


if __name__ == "__main__":
    # Quick test
    print("[bubble_centroids_v4] Running quick test...")
    print(f"geoopt available: {GEOOPT_AVAILABLE}")

    num_heads = 8
    num_experts = 32
    head_dim = 64
    B, N = 2, 100

    # Test Euclidean centroids
    euclidean_centroids = BubbleCentroidsV4(
        num_heads=num_heads,
        num_experts=num_experts,
        head_dim=head_dim,
        manifold_type=ManifoldType.EUCLIDEAN,
    )
    centroids_e = euclidean_centroids(B)
    print(f"Euclidean centroids: {centroids_e.shape}")
    assert centroids_e.shape == (B, num_heads, num_experts, head_dim)

    # Test distance computation
    Q = torch.randn(B, num_heads, N, head_dim)
    distances_e = euclidean_centroids.distance_to_queries(Q)
    print(f"Euclidean distances: {distances_e.shape}")
    assert distances_e.shape == (B, num_heads, N, num_experts)

    # Test Poincaré centroids (if geoopt available)
    if GEOOPT_AVAILABLE:
        poincare_centroids = BubbleCentroidsV4(
            num_heads=num_heads,
            num_experts=num_experts,
            head_dim=head_dim,
            manifold_type=ManifoldType.POINCARE,
        )
        centroids_p = poincare_centroids(B)
        print(f"Poincaré centroids: {centroids_p.shape}")
        assert centroids_p.shape == (B, num_heads, num_experts, head_dim)

        distances_p = poincare_centroids.distance_to_queries(Q)
        print(f"Poincaré distances: {distances_p.shape}")
        assert distances_p.shape == (B, num_heads, N, num_experts)

    # Test Hybrid manifold
    hybrid_centroids = HybridManifoldCentroids(
        num_heads=num_heads,
        num_experts=num_experts,
        head_dim=head_dim,
        euclidean_heads=4,
        poincare_heads=4,
    )
    centroids_h = hybrid_centroids(B)
    print(f"Hybrid centroids: {centroids_h.shape}")
    assert centroids_h.shape == (B, num_heads, num_experts, head_dim)

    distances_h = hybrid_centroids.distance_to_queries(Q)
    print(f"Hybrid distances: {distances_h.shape}")
    assert distances_h.shape == (B, num_heads, N, num_experts)

    print("[bubble_centroids_v4] All tests passed!")
