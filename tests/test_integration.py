"""Integration tests for the full Bubble Transformer pipeline."""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from plateau_attention import PlateauAttentionMechanism
from metrics import concentration_ratio, attention_entropy, compute_all_metrics


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def test_epsilon_sweep_produces_trend(self):
        """Sweeping epsilon should produce a monotonic trend in concentration."""
        B, N, D = 2, 32, 128
        x = torch.randn(B, N, D)

        results = []
        for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
            attn = PlateauAttentionMechanism(
                d_model=D, num_heads=4, epsilon=eps, tau_iters=10
            )
            _, attn_matrix = attn(x, return_attention=True)
            cr = concentration_ratio(attn_matrix)
            ent = attention_entropy(attn_matrix)
            results.append({"epsilon": eps, "cr": cr, "entropy": ent})

        # Verify trend: concentration ratio should increase with epsilon
        cr_values = [r["cr"] for r in results]
        self.assertLess(
            cr_values[0], cr_values[-1], f"CR should increase with eps: {cr_values}"
        )

    def test_metrics_stable_across_runs(self):
        """Metrics should be stable for the same input."""
        torch.manual_seed(42)
        embeddings = torch.randn(4, 32, 128)

        metrics1 = compute_all_metrics(embeddings)
        metrics2 = compute_all_metrics(embeddings)

        self.assertAlmostEqual(
            metrics1["effective_rank"], metrics2["effective_rank"], places=5
        )
        self.assertAlmostEqual(
            metrics1["anisotropy_index"], metrics2["anisotropy_index"], places=5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
