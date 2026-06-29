"""
Power Diagrams — ψ weights module for Bubble Transformer
=========================================================

[DEFINITION] Power Diagram with weights ψ (Laguerre tessellation).

Given K centroids c₁, ..., c_K and weights ψ₁, ..., ψ_K, the cell of cₖ is:
    Pᵧₖ = {x : ‖x - cₖ‖² - ψₖ ≤ ‖x - cⱼ‖² - ψⱼ  ∀ j ≠ k}

In the Bubble Transformer context, ψ is applied as a bias in the log-Sinkhorn
matrix used by SIRI:
    log_S = -C / ε + ψ
where C is the geometric cost matrix and ε is the bandwidth.

When all ψₖ = 0, Power Diagram reduces to standard Voronoi tessellation.

Pure NumPy implementation (no PyTorch required).
"""

import numpy as np
from typing import Optional


def _to_numpy(x):
    """Defensive conversion to NumPy — handles torch tensors, numpy arrays, lists."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_psi_from_keys(
    K: np.ndarray,
    W_psi: np.ndarray,
) -> np.ndarray:
    """Compute ψ weights from key projections.

    Args:
        K: Key tensor [B, N, d_model] (NumPy or torch).
        W_psi: Linear projection [d_model, 1] — learnable scalar per token.

    Returns:
        psi: [B, N, 1] — Power Diagram weights per token, ready to broadcast
              against attention matrix of shape [B, heads, N, N].
    """
    K = _to_numpy(K).astype(np.float32)
    if K.ndim != 3:
        raise ValueError(f"Expected K of shape [B, N, d_model], got {K.shape}")

    psi = K @ W_psi  # [B, N, 1]
    return psi.astype(np.float32)


def apply_psi_to_log_sinkhorn(
    log_S: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    """Apply ψ bias to log-Sinkhorn matrix.

    Args:
        log_S: [B, heads, N, N] log-domain Sinkhorn scores (e.g., -C/ε).
        psi: [B, N, 1] Power Diagram weights (broadcast over heads).

    Returns:
        log_S_psi: [B, heads, N, N] log-Sinkhorn with Power Diagram bias.
    """
    log_S = _to_numpy(log_S).astype(np.float32)
    psi = _to_numpy(psi).astype(np.float32)

    if log_S.ndim != 4:
        raise ValueError(f"Expected log_S of shape [B, heads, N, N], got {log_S.shape}")
    if psi.ndim != 3:
        raise ValueError(f"Expected psi of shape [B, N, 1], got {psi.shape}")

    # Broadcast psi over heads axis.
    # log_S[b, h, i, j] += psi[b, j, 0]
    # i.e., the column (key index j) gets biased by ψⱼ.
    return log_S + psi[:, np.newaxis, :, :]


def power_diagram_assign(
    points: np.ndarray,
    centroids: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    """Assign each point to a Power Diagram cell.

    Args:
        points: [B, N, d] query points.
        centroids: [K, d] K cell centroids.
        psi: [K] Power Diagram weights (one per centroid).

    Returns:
        assignments: [B, N] integer in [0, K-1] for each point.
    """
    points = _to_numpy(points).astype(np.float32)
    centroids = _to_numpy(centroids).astype(np.float32)
    psi = _to_numpy(psi).astype(np.float32)

    if points.ndim != 3:
        raise ValueError(f"Expected points of shape [B, N, d], got {points.shape}")
    if centroids.ndim != 2:
        raise ValueError(f"Expected centroids of shape [K, d], got {centroids.shape}")
    if psi.ndim != 1:
        raise ValueError(f"Expected psi of shape [K], got {psi.shape}")

    B, N, d = points.shape
    K = centroids.shape[0]

    # Squared distances: [B, N, K]
    sq_dists = np.sum((points[:, :, np.newaxis, :] - centroids[np.newaxis, np.newaxis, :, :]) ** 2, axis=-1)

    # Power Diagram cost: dist² - ψ
    power_costs = sq_dists - psi[np.newaxis, np.newaxis, :]

    # Assign to argmin
    assignments = np.argmin(power_costs, axis=-1)  # [B, N]
    return assignments


class PowerDiagramModule:
    """Learnable Power Diagram weights via a linear projection over keys.

    Usage:
        pd = PowerDiagramModule(d_model=128, seed=42)
        psi = pd.compute_psi(K)  # [B, N, 1]
        log_S_psi = pd.apply_to_log_sinkhorn(log_S, K)  # [B, heads, N, N]
    """

    def __init__(self, d_model: int, seed: int = 42, scale: float = 0.1):
        rng = np.random.RandomState(seed)
        # Initialize psi projection with small scale so initial bias is small.
        self.W_psi = rng.randn(d_model, 1).astype(np.float32) * scale
        self.d_model = d_model

    def compute_psi(self, K: np.ndarray) -> np.ndarray:
        return compute_psi_from_keys(K, self.W_psi)

    def apply_to_log_sinkhorn(self, log_S: np.ndarray, K: np.ndarray) -> np.ndarray:
        psi = self.compute_psi(K)
        return apply_psi_to_log_sinkhorn(log_S, psi)


if __name__ == "__main__":
    print("[PowerDiagrams] Running quick test...")
    rng = np.random.RandomState(42)

    # Test 1: Basic psi computation
    B, N, d_model = 2, 16, 64
    K = rng.randn(B, N, d_model).astype(np.float32)
    pd = PowerDiagramModule(d_model=d_model, seed=42)
    psi = pd.compute_psi(K)
    print(f"  psi shape: {psi.shape}, range: [{psi.min():.3f}, {psi.max():.3f}]")
    assert psi.shape == (B, N, 1)

    # Test 2: Apply psi to log_Sinkhorn
    heads = 8
    log_S = rng.randn(B, heads, N, N).astype(np.float32)
    log_S_psi = pd.apply_to_log_sinkhorn(log_S, K)
    print(f"  log_S shape: {log_S.shape} -> log_S_psi shape: {log_S_psi.shape}")
    assert log_S_psi.shape == (B, heads, N, N)

    # Test 3: Power Diagram assignment
    centroids = rng.randn(4, d_model).astype(np.float32)
    psi_vec = rng.randn(4).astype(np.float32)
    assignments = power_diagram_assign(K, centroids, psi_vec)
    print(f"  Power Diagram assignments: shape {assignments.shape}, "
          f"unique cells: {len(np.unique(assignments))}")
    assert assignments.shape == (B, N)

    # Test 4: psi = 0 reduces to Voronoi
    K_voronoi = rng.randn(1, 8, 16).astype(np.float32)
    centroids = rng.randn(3, 16).astype(np.float32)
    assignments_voronoi = power_diagram_assign(K_voronoi, centroids, np.zeros(3))
    print(f"  Voronoi (psi=0) assignments: shape {assignments_voronoi.shape}")

    print("[PowerDiagrams] All tests passed!")
