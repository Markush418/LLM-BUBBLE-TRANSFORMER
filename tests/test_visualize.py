"""Unit tests for experiments/visualize.py.

Tests all plot generation functions with mock data, verifying that PNG
files are created correctly.  Uses a temporary directory for plot output
to avoid polluting the workspace.
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from test_helpers import create_mock_embeddings, mock_matplotlib_figure


@unittest.skipIf(
    __import__("importlib").util.find_spec("matplotlib") is None,
    "matplotlib not installed",
)
class TestVisualize(unittest.TestCase):
    """Test suite for visualize.py plot generators."""

    def setUp(self):
        """Redirect plot output to a temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        self.plot_dir = Path(self.test_dir) / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Patch the module-level PLOT_DIR so _save_plot writes to temp dir
        import visualize

        self._orig_plot_dir = visualize.PLOT_DIR
        visualize.PLOT_DIR = self.plot_dir

    def tearDown(self):
        """Restore original PLOT_DIR and clean up temp files."""
        import visualize

        visualize.PLOT_DIR = self._orig_plot_dir
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_mock_results(self, num_layers=3, num_epsilons=3):
        """Build a list of result dicts mimicking epsilon-sweep output."""
        layers = [3, 7, 11][:num_layers]
        epsilons = [0.001, 0.01, 0.1][:num_epsilons]
        results = []
        for layer in layers:
            for eps in epsilons:
                results.append(
                    {
                        "layer": layer,
                        "epsilon": eps,
                        "effective_rank": 100.0 + eps * 50.0,
                        "concentration_ratio": 0.1 + eps * 0.2,
                        "anisotropy_index": 0.3 + eps * 0.1,
                        "intrinsic_dim_mle": 50.0 + layer * 5.0,
                        "cost_type": "l2_sq",
                    }
                )
        return results

    def _create_baseline_ranks(self):
        return {3: 150.0, 7: 140.0, 11: 130.0}

    def _create_sweet_spot(self):
        return {
            "epsilon": 0.01,
            "layers": [3, 7],
            "concentration_ratio": 0.25,
            "effective_rank": 120.0,
            "intrinsic_dim_mle": 60.0,
            "confidence": "medium",
        }

    def _create_mock_embeddings_dict(self):
        """Return {layer: np.ndarray} for t-SNE tests."""
        rng = np.random.RandomState(42)
        return {
            3: rng.randn(50, 128).astype(np.float32),
            7: rng.randn(50, 128).astype(np.float32),
            11: rng.randn(50, 128).astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Core plot tests
    # ------------------------------------------------------------------

    def test_plot_effective_rank_curves(self):
        """effective_rank_curves.png must be created."""
        from visualize import plot_effective_rank_curves

        results = self._create_mock_results()
        baseline = self._create_baseline_ranks()
        plot_effective_rank_curves(results, baseline)

        path = self.plot_dir / "effective_rank_curves.png"
        self.assertTrue(path.exists(), f"Missing plot: {path.name}")
        self.assertGreater(path.stat().st_size, 0)

    def test_plot_concentration_heatmap(self):
        """concentration_heatmap_*.png must be created."""
        from visualize import plot_concentration_heatmap

        results = self._create_mock_results()
        plot_concentration_heatmap(results, metric="concentration_ratio")

        path = self.plot_dir / "concentration_heatmap_concentration_ratio.png"
        self.assertTrue(path.exists(), f"Missing plot: {path.name}")

    def test_plot_pareto_frontier(self):
        """pareto_frontier.png must be created."""
        from visualize import plot_pareto_frontier

        results = self._create_mock_results()
        plot_pareto_frontier(results)

        path = self.plot_dir / "pareto_frontier.png"
        self.assertTrue(path.exists(), f"Missing plot: {path.name}")

    def test_plot_anisotropy_vs_epsilon(self):
        """anisotropy_vs_epsilon.png must be created."""
        from visualize import plot_anisotropy_vs_epsilon

        results = self._create_mock_results()
        plot_anisotropy_vs_epsilon(results)

        path = self.plot_dir / "anisotropy_vs_epsilon.png"
        self.assertTrue(path.exists(), f"Missing plot: {path.name}")

    def test_plot_intrinsic_dim_vs_epsilon(self):
        """intrinsic_dim_vs_epsilon.png must be created."""
        from visualize import plot_intrinsic_dim_vs_epsilon

        results = self._create_mock_results()
        plot_intrinsic_dim_vs_epsilon(results)

        path = self.plot_dir / "intrinsic_dim_vs_epsilon.png"
        self.assertTrue(path.exists(), f"Missing plot: {path.name}")

    @unittest.skipIf(
        __import__("importlib").util.find_spec("sklearn") is None,
        "sklearn not installed",
    )
    def test_plot_tsne_embeddings(self):
        """tsne_layer_*.png files must be created for requested layers."""
        from visualize import plot_tsne_embeddings

        embeddings = self._create_mock_embeddings_dict()
        plot_tsne_embeddings(embeddings, layer_indices=[3, 7])

        for layer in [3, 7]:
            path = self.plot_dir / f"tsne_layer_{layer}.png"
            self.assertTrue(path.exists(), f"Missing plot: {path.name}")

    def test_generate_summary_dashboard(self):
        """summary_dashboard.png must be created."""
        from visualize import generate_summary_dashboard

        results = self._create_mock_results()
        sweet_spot = self._create_sweet_spot()
        generate_summary_dashboard(results, sweet_spot)

        path = self.plot_dir / "summary_dashboard.png"
        self.assertTrue(path.exists(), f"Missing plot: {path.name}")

    def test_plot_cost_comparison_pareto(self):
        """cost_comparison_pareto.png must be created."""
        from visualize import plot_cost_comparison_pareto

        results = self._create_mock_results()
        plot_cost_comparison_pareto(results)

        path = self.plot_dir / "cost_comparison_pareto.png"
        self.assertTrue(path.exists(), f"Missing plot: {path.name}")

    def test_plot_tension_analysis(self):
        """Tension analysis must produce 4 PNG files from JSON input."""
        from visualize import plot_tension_analysis

        tension_data = {
            "results": [
                {
                    "layer": 3,
                    "alpha": 0.0,
                    "effective_rank": 100,
                    "concentration_ratio": 0.2,
                    "tension_balance": 0.5,
                },
                {
                    "layer": 3,
                    "alpha": 0.5,
                    "effective_rank": 120,
                    "concentration_ratio": 0.15,
                    "tension_balance": 0.3,
                },
                {
                    "layer": 7,
                    "alpha": 0.0,
                    "effective_rank": 110,
                    "concentration_ratio": 0.25,
                    "tension_balance": 0.6,
                },
            ],
            "baseline_ranks": {"3": 150, "7": 140},
            "optimal_alpha": {"alpha": 0.5, "layer": 3},
        }
        tension_path = Path(self.test_dir) / "tension_sweep.json"
        with open(tension_path, "w") as f:
            json.dump(tension_data, f)

        plot_tension_analysis(str(tension_path))

        expected = [
            "tension_alpha_vs_rank.png",
            "tension_alpha_vs_concentration.png",
            "tension_pareto.png",
            "tension_balance_heatmap.png",
        ]
        for name in expected:
            self.assertTrue(
                (self.plot_dir / name).exists(),
                f"Missing plot: {name}",
            )

    def test_plot_layer_selection(self):
        """Layer selection must produce 5 PNG files from JSON input."""
        from visualize import plot_layer_selection

        ls_data = {
            "baseline": {
                "3": {
                    "effective_rank": 150,
                    "concentration_ratio": 0.2,
                    "intrinsic_dim_mle": 60,
                },
                "7": {
                    "effective_rank": 140,
                    "concentration_ratio": 0.25,
                    "intrinsic_dim_mle": 55,
                },
            },
            "plateau": {
                "3": {
                    "effective_rank": 145,
                    "concentration_ratio": 0.15,
                    "intrinsic_dim_mle": 58,
                },
                "7": {
                    "effective_rank": 138,
                    "concentration_ratio": 0.18,
                    "intrinsic_dim_mle": 54,
                },
            },
            "ranked_layers": [3, 7],
            "layer_scores": {
                "3": {"score": 0.85},
                "7": {"score": 0.75},
            },
        }
        ls_path = Path(self.test_dir) / "layer_selection.json"
        with open(ls_path, "w") as f:
            json.dump(ls_data, f)

        plot_layer_selection(str(ls_path))

        expected = [
            "layer_selection_rank_comparison.png",
            "layer_selection_concentration_gain.png",
            "layer_selection_intrinsic_dim_preservation.png",
            "layer_selection_pareto_movement.png",
            "layer_selection_ranking.png",
        ]
        for name in expected:
            self.assertTrue(
                (self.plot_dir / name).exists(),
                f"Missing plot: {name}",
            )

    def test_generate_all_plots(self):
        """Orchestrator generate_all_plots must create all 8 expected PNGs."""
        from visualize import generate_all_plots

        sweep_data = {
            "results": self._create_mock_results(),
            "baseline_ranks": self._create_baseline_ranks(),
            "sweet_spot": self._create_sweet_spot(),
        }
        sweep_path = Path(self.test_dir) / "epsilon_sweep.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep_data, f)

        generate_all_plots(str(sweep_path))

        expected = [
            "effective_rank_curves.png",
            "concentration_heatmap_concentration_ratio.png",
            "concentration_heatmap_effective_rank.png",
            "pareto_frontier.png",
            "cost_comparison_pareto.png",
            "anisotropy_vs_epsilon.png",
            "intrinsic_dim_vs_epsilon.png",
            "summary_dashboard.png",
        ]
        for name in expected:
            self.assertTrue(
                (self.plot_dir / name).exists(),
                f"Missing plot: {name}",
            )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_results(self):
        """Plot functions must not raise with empty result lists."""
        from visualize import plot_effective_rank_curves, plot_pareto_frontier

        plot_effective_rank_curves([], {})
        self.assertTrue((self.plot_dir / "effective_rank_curves.png").exists())

        plot_pareto_frontier([])
        self.assertTrue((self.plot_dir / "pareto_frontier.png").exists())

    def test_single_point_results(self):
        """Plot functions must work with a single data point."""
        from visualize import plot_concentration_heatmap, plot_anisotropy_vs_epsilon

        results = [
            {
                "layer": 3,
                "epsilon": 0.01,
                "effective_rank": 100.0,
                "concentration_ratio": 0.2,
                "anisotropy_index": 0.3,
                "intrinsic_dim_mle": 50.0,
            }
        ]

        plot_concentration_heatmap(results)
        self.assertTrue(
            (self.plot_dir / "concentration_heatmap_concentration_ratio.png").exists()
        )

        plot_anisotropy_vs_epsilon(results)
        self.assertTrue((self.plot_dir / "anisotropy_vs_epsilon.png").exists())

    def test_missing_json_file_returns_early(self):
        """File-based plotters must return silently when JSON is missing."""
        from visualize import plot_tension_analysis, plot_layer_selection

        missing = str(Path(self.test_dir) / "does_not_exist.json")
        # Should not raise
        plot_tension_analysis(missing)
        plot_layer_selection(missing)

    def test_summary_dashboard_with_zero_values(self):
        """Dashboard must work when sweet_spot contains zero/edge values."""
        from visualize import generate_summary_dashboard

        results = self._create_mock_results()
        sweet_spot = {
            "epsilon": 0.001,
            "layers": [3],
            "concentration_ratio": 0.0,
            "effective_rank": 0.0,
            "intrinsic_dim_mle": 0.0,
            "confidence": "low",
        }
        generate_summary_dashboard(results, sweet_spot)
        self.assertTrue((self.plot_dir / "summary_dashboard.png").exists())

    def test_cost_comparison_multiple_cost_types(self):
        """Cost comparison must handle mixed cost_type entries."""
        from visualize import plot_cost_comparison_pareto

        results = self._create_mock_results(num_layers=3, num_epsilons=3)
        for i, r in enumerate(results):
            r["cost_type"] = ["l2_sq", "cosine", "dot_product"][i % 3]

        plot_cost_comparison_pareto(results)
        self.assertTrue((self.plot_dir / "cost_comparison_pareto.png").exists())

    # ------------------------------------------------------------------
    # Infrastructure / helper tests
    # ------------------------------------------------------------------

    def test_matplotlib_agg_backend(self):
        """Verify that the non-interactive Agg backend is active."""
        import matplotlib

        self.assertEqual(matplotlib.get_backend().lower(), "agg")

    def test_mock_matplotlib_figure_from_helpers(self):
        """mock_matplotlib_figure must return a usable figure object."""
        fig = mock_matplotlib_figure()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, "savefig"))
        self.assertTrue(hasattr(fig, "add_subplot"))

    def test_create_mock_embeddings_adapted_for_tsne(self):
        """create_mock_embeddings can be adapted to numpy for t-SNE input."""
        emb_torch = create_mock_embeddings(1, 50, 128)
        emb_np = emb_torch.numpy().squeeze(0)
        self.assertEqual(emb_np.shape, (50, 128))
        self.assertEqual(emb_np.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
