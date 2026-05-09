"""Comprehensive unit tests for the epsilon sweep module.

Tests cover sweep execution, sweet spot identification, edge cases,
and monotonic trends using purely mock data (no model downloads).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from epsilon_sweep import identify_sweet_spot, run_epsilon_sweep
from test_helpers import create_mock_embeddings


class TestEpsilonSweep(unittest.TestCase):
    """Tests for ``run_epsilon_sweep()`` and ``identify_sweet_spot()``.

    All tests use synthetic embeddings written to temporary directories so
    that no real model extraction or network access is required.
    """

    def setUp(self):
        """Create a temporary directory hierarchy for mock embeddings."""
        self.B = 2
        self.N = 16
        self.D = 64
        self.num_heads = 4
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings_dir = os.path.join(self.temp_dir, "embeddings")
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(os.path.join(self.embeddings_dir, "softmax"), exist_ok=True)

    def tearDown(self):
        """Remove the temporary directory and all contents."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _save_mock_embeddings(self, d_model=None):
        """Write ``raw_input.npy`` and ``softmax/layer_*.npy`` files.

        Parameters
        ----------
        d_model : int, optional
            Embedding dimension.  Defaults to ``self.D``.
        """
        d_model = d_model or self.D
        raw = create_mock_embeddings(self.B, self.N, d_model).numpy().astype(np.float32)
        np.save(os.path.join(self.embeddings_dir, "raw_input.npy"), raw)
        for layer_idx in [3, 7]:
            emb = (
                create_mock_embeddings(self.B, self.N, d_model)
                .numpy()
                .astype(np.float32)
            )
            np.save(
                os.path.join(self.embeddings_dir, "softmax", f"layer_{layer_idx}.npy"),
                emb,
            )

    def test_run_epsilon_sweep_basic(self):
        """Basic sweep should return a dict with expected top-level keys."""
        self._save_mock_embeddings()
        result = run_epsilon_sweep(
            embeddings_dir=self.embeddings_dir,
            output_dir=self.results_dir,
            epsilon_values=[0.01, 0.1],
            target_layers=[3],
            d_model=self.D,
            num_heads=self.num_heads,
        )
        self.assertIn("results", result)
        self.assertIn("sweet_spot", result)
        self.assertIn("config", result)
        self.assertIn("baseline_ranks", result)
        self.assertGreater(len(result["results"]), 0)

    def test_run_epsilon_sweep_no_raw_input(self):
        """Missing ``raw_input.npy`` should cause an early return with an empty dict."""
        result = run_epsilon_sweep(
            embeddings_dir=self.embeddings_dir,
            output_dir=self.results_dir,
            epsilon_values=[0.01],
            target_layers=[3],
            d_model=self.D,
            num_heads=self.num_heads,
        )
        self.assertEqual(result, {})

    def test_run_epsilon_sweep_creates_output_file(self):
        """A successful sweep must write ``epsilon_sweep.json`` to the output dir."""
        self._save_mock_embeddings()
        run_epsilon_sweep(
            embeddings_dir=self.embeddings_dir,
            output_dir=self.results_dir,
            epsilon_values=[0.05],
            target_layers=[3],
            d_model=self.D,
            num_heads=self.num_heads,
        )
        output_path = os.path.join(self.results_dir, "epsilon_sweep.json")
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r") as f:
            data = json.load(f)
        self.assertIn("sweet_spot", data)
        self.assertIn("results", data)
        self.assertIn("config", data)

    def test_concentration_ratio_monotonic_trend(self):
        """Concentration ratio should increase with epsilon (first < last).

        Smaller epsilon produces sparser attention (lower CR) while larger
        epsilon approaches uniform attention (higher CR).
        """
        self._save_mock_embeddings()
        result = run_epsilon_sweep(
            embeddings_dir=self.embeddings_dir,
            output_dir=self.results_dir,
            epsilon_values=[0.01, 0.05, 0.1, 0.5, 1.0],
            target_layers=[3],
            d_model=self.D,
            num_heads=self.num_heads,
        )
        plateau_results = [
            r
            for r in result["results"]
            if r.get("epsilon", 0) > 0 and r.get("layer") == 3 and "error" not in r
        ]
        self.assertGreaterEqual(len(plateau_results), 2)
        cr_values = [r["concentration_ratio"] for r in plateau_results]
        self.assertLess(
            cr_values[0],
            cr_values[-1],
            f"CR should increase with epsilon: {cr_values}",
        )

    def test_run_epsilon_sweep_extreme_epsilon_values(self):
        """Extreme epsilon values (0.0 and 100.0) should be handled gracefully.

        epsilon=0.0 triggers a division-by-zero in Sinkhorn and must be caught
        as an error entry rather than crashing the sweep.
        """
        self._save_mock_embeddings()
        result = run_epsilon_sweep(
            embeddings_dir=self.embeddings_dir,
            output_dir=self.results_dir,
            epsilon_values=[0.0, 100.0],
            target_layers=[3],
            d_model=self.D,
            num_heads=self.num_heads,
        )
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)
        # At least one entry should be an error or a valid result; the sweep
        # must not raise an unhandled exception.
        for r in result["results"]:
            self.assertIn("layer", r)
            self.assertIn("epsilon", r)

    def test_identify_sweet_spot_with_known_data(self):
        """``identify_sweet_spot`` should pick the epsilon with best concentration
        among those that satisfy the geometric constraints.
        """
        results = []
        for eps in [0.01, 0.05, 0.1]:
            for layer in [3]:
                results.append(
                    {
                        "layer": layer,
                        "epsilon": eps,
                        "concentration_ratio": 0.1 if eps == 0.05 else 0.3,
                        "effective_rank": 80.0,
                        "anisotropy_index": 0.3,
                        "intrinsic_dim_mle": 30.0,
                    }
                )
        sweet_spot = identify_sweet_spot(results, [0.01, 0.05, 0.1], [3])
        self.assertEqual(sweet_spot["epsilon"], 0.05)
        self.assertIn("layers", sweet_spot)
        self.assertIn("confidence", sweet_spot)

    def test_identify_sweet_spot_empty_results(self):
        """Empty results should return ``epsilon=None`` with an explanatory reason."""
        sweet_spot = identify_sweet_spot([], [0.01, 0.1], [3])
        self.assertIsNone(sweet_spot["epsilon"])
        self.assertEqual(sweet_spot["reason"], "No valid results")

    def test_identify_sweet_spot_single_epsilon(self):
        """A single epsilon that passes constraints should be selected as the sweet spot."""
        results = [
            {
                "layer": 3,
                "epsilon": 0.5,
                "concentration_ratio": 0.2,
                "effective_rank": 100.0,
                "anisotropy_index": 0.1,
                "intrinsic_dim_mle": 50.0,
            }
        ]
        sweet_spot = identify_sweet_spot(results, [0.5], [3])
        self.assertEqual(sweet_spot["epsilon"], 0.5)
        self.assertEqual(sweet_spot["confidence"], "high")

    def test_identify_sweet_spot_all_errors(self):
        """If every result contains an error, no sweet spot can be identified."""
        results = [
            {"layer": 3, "epsilon": 0.1, "error": "RuntimeError"},
            {"layer": 3, "epsilon": 0.5, "error": "ValueError"},
        ]
        sweet_spot = identify_sweet_spot(results, [0.1, 0.5], [3])
        self.assertIsNone(sweet_spot["epsilon"])
        self.assertEqual(sweet_spot["reason"], "No valid results")

    def test_identify_sweet_spot_extreme_constraints(self):
        """Boundary values for rank, anisotropy, and intrinsic-dim constraints.

        - Exactly at boundaries (rank=50, aniso=0.49, dim=20) → high confidence.
        - Just below boundaries (rank=49.9, aniso=0.51, dim=19.9) → medium confidence.
        """
        # Pass case
        results_pass = [
            {
                "layer": 3,
                "epsilon": 0.1,
                "concentration_ratio": 0.2,
                "effective_rank": 50.0,
                "anisotropy_index": 0.49,
                "intrinsic_dim_mle": 20.0,
            }
        ]
        sweet_spot = identify_sweet_spot(results_pass, [0.1], [3])
        self.assertEqual(sweet_spot["epsilon"], 0.1)
        self.assertEqual(sweet_spot["confidence"], "high")

        # Fail case
        results_fail = [
            {
                "layer": 3,
                "epsilon": 0.1,
                "concentration_ratio": 0.2,
                "effective_rank": 49.9,
                "anisotropy_index": 0.51,
                "intrinsic_dim_mle": 19.9,
            }
        ]
        sweet_spot = identify_sweet_spot(results_fail, [0.1], [3])
        self.assertEqual(sweet_spot["epsilon"], 0.1)
        self.assertEqual(sweet_spot["confidence"], "medium")

    def test_run_epsilon_sweep_with_metadata(self):
        """``run_epsilon_sweep`` should auto-detect ``d_model`` and ``num_heads``
        from ``metadata.json`` when the parameters are not supplied explicitly.
        """
        self._save_mock_embeddings(d_model=128)
        metadata = {
            "d_model": 128,
            "num_attention_heads": 8,
            "mode": "mock_numpy",
        }
        with open(os.path.join(self.embeddings_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        result = run_epsilon_sweep(
            embeddings_dir=self.embeddings_dir,
            output_dir=self.results_dir,
            epsilon_values=[0.1],
            target_layers=[3],
        )
        self.assertEqual(result["config"]["d_model"], 128)
        self.assertEqual(result["config"]["num_heads"], 8)
        self.assertEqual(result["mode"], "mock_numpy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
