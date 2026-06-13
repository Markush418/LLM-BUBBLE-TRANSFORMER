"""
Baroreceptor MLP — Dynamic Centroid Prediction
==============================================

Predicts the optimal number of centroids C based on input variance.
Analogous to biological baroreceptors that regulate blood pressure.

Architecture: d_model → 64 → 1 → sigmoid → C range [min_C, max_C]
"""

import torch
import torch.nn as nn
from typing import Tuple


class BaroreceptorMLP(nn.Module):
    """
    Predicts optimal number of centroids C based on input variance.

    Analogía biológica: Los barorreceptores regulan la presión arterial.
    Este MLP "regula la presión" del espacio de representación.

    Args:
        d_model: Input dimension
        min_C: Minimum number of centroids (default: 16)
        max_C: Maximum number of centroids (default: 512)
        use_alpha_prediction: Whether to enable learned alpha prediction head (default: False)
    """

    def __init__(self, d_model: int, min_C: int = 16, max_C: int = 512, use_alpha_prediction: bool = False):
        super().__init__()
        self.min_C = min_C
        self.max_C = max_C
        self.use_alpha_prediction = use_alpha_prediction

        # Ultra-lightweight MLP for C prediction
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # Optional alpha prediction head
        if use_alpha_prediction:
            self.alpha_net = nn.Sequential(
                nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> int:
        """
        Predict number of centroids C.

        Args:
            x: [B, N, d_model] - input embeddings

        Returns:
            C: int - number of centroids (in range [min_C, max_C])
        """
        # Pool over sequence dimension
        x_pooled = x.mean(dim=1)  # [B, d_model]

        # Predict pressure (normalized to [0, 1])
        pressure = self.net(x_pooled)  # [B, 1]

        # Map to C range
        C = self.min_C + pressure * (self.max_C - self.min_C)

        # Return as integer (take first batch item)
        return int(C[0].round().item())

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict C for each batch item (for batch processing).

        Args:
            x: [B, N, d_model]

        Returns:
            C: [B] - number of centroids per batch item
        """
        x_pooled = x.mean(dim=1)  # [B, d_model]
        pressure = self.net(x_pooled).squeeze(-1)  # [B]
        C = self.min_C + pressure * (self.max_C - self.min_C)
        return C.round().int()

    def predict_alpha(self, x: torch.Tensor) -> float:
        """
        Predict tension coefficient alpha based on input variance.

        Maps input variance to alpha in [0.3, 0.8] using a sigmoid-based
        smooth transition. Higher variance inputs tend toward lower alpha
        (more expressivity), while lower variance inputs tend toward higher
        alpha (more concentration).

        Args:
            x: [B, N, d_model] - input embeddings

        Returns:
            alpha: float in range [0.3, 0.8]
        """
        # Compute input variance across sequence dimension
        var = x.var(dim=1).mean()  # scalar

        # Threshold and scale for sigmoid mapping
        threshold = 1.0
        scale = 2.0

        # Map variance to [0, 1] via sigmoid (inverted so high var -> low alpha)
        alpha_raw = torch.sigmoid((threshold - var) * scale)

        # Remap to [0.3, 0.8]
        alpha = 0.3 + alpha_raw * 0.5

        return alpha.item()

    def forward_with_alpha(self, x: torch.Tensor) -> Tuple[int, float]:
        """
        Predict both number of centroids C and tension coefficient alpha.

        Args:
            x: [B, N, d_model] - input embeddings

        Returns:
            (C, alpha): C is int in [min_C, max_C], alpha is float in [0.3, 0.8]
        """
        C = self.forward(x)

        if self.use_alpha_prediction and hasattr(self, 'alpha_net'):
            # Use MLP head for alpha prediction from pooled embeddings
            x_pooled = x.mean(dim=1)  # [B, d_model]
            alpha_raw = self.alpha_net(x_pooled)  # [B, 1]
            alpha = 0.3 + alpha_raw[0].item() * 0.5
        else:
            # Fall back to variance-based prediction
            alpha = self.predict_alpha(x)

        return C, alpha


if __name__ == "__main__":
    # Quick test
    print("[baroreceptor] Running quick test...")

    B, N, d_model = 4, 128, 512
    x = torch.randn(B, N, d_model)

    baroreceptor = BaroreceptorMLP(d_model=d_model, min_C=16, max_C=512)

    # Test single prediction
    C = baroreceptor(x)
    print(f"Input: {x.shape} -> C={C}")
    assert 16 <= C <= 512, f"C out of range: {C}"

    # Test batch prediction
    C_batch = baroreceptor.forward_batch(x)
    print(f"Batch predictions: {C_batch}")
    assert C_batch.shape == (B,), f"Expected {(B,)}, got {C_batch.shape}"
    assert (C_batch >= 16).all() and (C_batch <= 512).all(), "Batch C out of range"

    # Test with different variance inputs
    x_low_var = torch.randn(B, N, d_model) * 0.1  # Low variance
    x_high_var = torch.randn(B, N, d_model) * 10.0  # High variance

    C_low = baroreceptor(x_low_var)
    C_high = baroreceptor(x_high_var)
    print(f"Low variance input -> C={C_low}")
    print(f"High variance input -> C={C_high}")

    print("[baroreceptor] All tests passed!")
