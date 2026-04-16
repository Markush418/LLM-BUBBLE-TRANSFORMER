"""
Layer Selection Tests — Plan D
===============================
Test the new layer selection functionality.
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

# Add experiments to path for sibling imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from config import get_config, LayerSelectionConfig
from metrics import compute_all_metrics
from epsilon_sweep import run_layer_selection
from visualize import plot_layer_selection


class TestLayerSelectionConfig(unittest.TestCase):
    """Test LayerSelectionConfig dataclass."""

    def test_default_values(self):
        cfg = LayerSelectionConfig()
        self.assertTrue(cfg.enable_dual_head)
        self.assertEqual(len(cfg.alpha_values), 11)
        self.assertIn("effective_rank", cfg.comparison_metrics)

    def test_custom_values(self):
        cfg = LayerSelectionConfig(
            enable_dual_head=False,
            alpha_values=[0.0, 0.5, 1.0],
            comparison_metrics=["concentration_ratio"],
        )
        self.assertFalse(cfg.enable_dual_head)
        self.assertEqual(cfg.alpha_values, [0.0, 0.5, 1.0])
        self.assertEqual(cfg.comparison_metrics, ["concentration_ratio"])


class TestRunLayerSelection(unittest.TestCase):
    """Test the run_layer_selection function with mock embeddings."""

    @classmethod
    def setUpClass(cls):
        """Create temporary mock embeddings for testing."""
        cls.test_dir = tempfile.mkdtemp()
        cls.embeddings_dir = Path(cls.test_dir) / "embeddings"
        cls.results_dir = Path(cls.test_dir) / "results"
        cls.embeddings_dir.mkdir(parents=True)
        cls.results_dir.mkdir(parents=True)

        # Create metadata
        metadata = {
            "mode": "mock",
            "d_model": 512,
            "num_attention_heads": 8,
        }
        with open(cls.embeddings_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create raw_input.npy
        raw = np.random.randn(4, 64, 512).astype(np.float32)
        np.save(cls.embeddings_dir / "raw_input.npy", raw)

        # Create softmax embeddings for layers 3, 7, 11
        softmax_dir = cls.embeddings_dir / "softmax"
        softmax_dir.mkdir(parents=True)
        for layer in [3, 7, 11]:
            emb = np.random.randn(4, 64, 512).astype(np.float32)
            np.save(softmax_dir / f"layer_{layer}.npy", emb)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.test_dir)

    def test_run_layer_selection_returns_dict(self):
        """Test that run_layer_selection returns a dictionary."""
        results = run_layer_selection(
            embeddings_dir=str(self.embeddings_dir),
            output_dir=str(self.results_dir),
            d_model=512,
            num_heads=8,
            cost_type="l2_sq",
            epsilon_plateau=0.001,
        )
        self.assertIsInstance(results, dict)
        self.assertIn("baseline", results)
        self.assertIn("plateau", results)
        self.assertIn("ranked_layers", results)

    def test_layer_selection_results_have_all_layers(self):
        """Test that results contain entries for all target layers."""
        results = run_layer_selection(
            embeddings_dir=str(self.embeddings_dir),
            output_dir=str(self.results_dir),
            d_model=512,
            num_heads=8,
            cost_type="l2_sq",
            epsilon_plateau=0.001,
        )
        baseline = results["baseline"]
        plateau = results["plateau"]
        for layer in [3, 7, 11]:
            self.assertIn(layer, baseline)
            self.assertIn(layer, plateau)
            # Check that metrics exist
            self.assertIn("effective_rank", baseline[layer])
            self.assertIn("effective_rank", plateau[layer])

    def test_layer_selection_metrics_are_valid(self):
        """Test that metrics have reasonable values."""
        results = run_layer_selection(
            embeddings_dir=str(self.embeddings_dir),
            output_dir=str(self.results_dir),
            d_model=512,
            num_heads=8,
            cost_type="l2_sq",
            epsilon_plateau=0.001,
        )
        baseline = results["baseline"]
        plateau = results["plateau"]

        for layer in [3, 7, 11]:
            b = baseline[layer]
            p = plateau[layer]
            # Effective rank should be positive
            self.assertGreater(b["effective_rank"], 0)
            self.assertGreater(p["effective_rank"], 0)
            # Concentration ratio should be in [0, 1]
            self.assertGreaterEqual(b.get("concentration_ratio", 0), 0)
            self.assertLessEqual(b.get("concentration_ratio", 1), 1)
            self.assertGreaterEqual(p.get("concentration_ratio", 0), 0)
            self.assertLessEqual(p.get("concentration_ratio", 1), 1)

    def test_plot_layer_selection_creates_files(self):
        """Test that plot_layer_selection generates PNG files."""
        # First, generate a layer_selection.json
        results = run_layer_selection(
            embeddings_dir=str(self.embeddings_dir),
            output_dir=str(self.results_dir),
            d_model=512,
            num_heads=8,
            cost_type="l2_sq",
            epsilon_plateau=0.001,
        )
        json_path = self.results_dir / "layer_selection.json"
        self.assertTrue(json_path.exists())

        # Call plot function
        plot_layer_selection(str(json_path))
        # Check that plot files exist
        expected_plots = [
            "layer_selection_rank_comparison.png",
            "layer_selection_concentration_gain.png",
            "layer_selection_intrinsic_dim_preservation.png",
            "layer_selection_pareto_movement.png",
            "layer_selection_ranking.png",
        ]
        for name in expected_plots:
            plot_path = Path("plots") / name
            self.assertTrue(plot_path.exists(), f"Missing plot: {name}")

    def test_layer_selection_with_dual_head(self):
        """Test layer selection with dual-head enabled."""
        results = run_layer_selection(
            embeddings_dir=str(self.embeddings_dir),
            output_dir=str(self.results_dir),
            d_model=512,
            num_heads=8,
            cost_type="l2_sq",
            epsilon_plateau=0.001,
            config=None,  # Uses internal default: enable_dual_head=True
        )
        # dual_head should be populated
        self.assertIsNotNone(results.get("dual_head"))
        # Optimal alpha may be None if all dual runs fail, but structure exists
        self.assertIn("optimal_alpha", results)

    def test_ranked_layers_ordering(self):
        """Test that ranked_layers are sorted by score descending."""
        results = run_layer_selection(
            embeddings_dir=str(self.embeddings_dir),
            output_dir=str(self.results_dir),
            d_model=512,
            num_heads=8,
            cost_type="l2_sq",
            epsilon_plateau=0.001,
        )
        ranked = results["ranked_layers"]
        scores = [results["layer_scores"][l]["score"] for l in ranked]
        # Check that scores are non-increasing
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])


if __name__ == "__main__":
    unittest.main()
