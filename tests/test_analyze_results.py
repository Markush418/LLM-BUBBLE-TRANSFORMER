"""Tests for the post-experiment analysis module."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from analyze_results import (
    export_optimal_config,
    generate_report,
    layer_recommendations,
    load_results,
    statistical_comparison,
)


class TestAnalyzeResults(unittest.TestCase):
    """Comprehensive tests for analyze_results.py functions."""

    def setUp(self):
        """Create a temporary directory for mock JSON files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_results_path = os.path.join(self.temp_dir.name, "mock_results.json")

        # Build a minimal but representative synthetic dataset
        self.mock_data = {
            "experiment": "Test Experiment",
            "date": "2026-05-09",
            "mode": "mock",
            "config": {
                "epsilon_values": [0.001, 0.01, 0.1],
                "target_layers": [3, 7, 11],
            },
            "baseline_ranks": {
                "3": 100.0,
                "7": 90.0,
                "11": 80.0,
            },
            "sweet_spot": {
                "epsilon": 0.01,
                "layers": [3, 7],
                "concentration_ratio": 0.35,
                "effective_rank": 85.0,
                "anisotropy_index": 0.05,
                "intrinsic_dim_mle": 25.0,
                "confidence": "high",
            },
            "results": [
                # Layer 3, epsilon 0.001 — too collapsed (low effective rank)
                {
                    "layer": 3,
                    "epsilon": 0.001,
                    "effective_rank": 30.0,
                    "concentration_ratio": 0.25,
                    "anisotropy_index": 0.02,
                },
                # Layer 3, epsilon 0.01 — good (within constraints)
                {
                    "layer": 3,
                    "epsilon": 0.01,
                    "effective_rank": 85.0,
                    "concentration_ratio": 0.35,
                    "anisotropy_index": 0.05,
                },
                # Layer 3, epsilon 0.1 — too anisotropic
                {
                    "layer": 3,
                    "epsilon": 0.1,
                    "effective_rank": 95.0,
                    "concentration_ratio": 0.55,
                    "anisotropy_index": 0.60,
                },
                # Layer 7, epsilon 0.001 — collapsed
                {
                    "layer": 7,
                    "epsilon": 0.001,
                    "effective_rank": 20.0,
                    "concentration_ratio": 0.20,
                    "anisotropy_index": 0.03,
                },
                # Layer 7, epsilon 0.01 — best (lowest concentration ratio)
                {
                    "layer": 7,
                    "epsilon": 0.01,
                    "effective_rank": 80.0,
                    "concentration_ratio": 0.30,
                    "anisotropy_index": 0.04,
                },
                # Layer 7, epsilon 0.1 — acceptable but worse score
                {
                    "layer": 7,
                    "epsilon": 0.1,
                    "effective_rank": 88.0,
                    "concentration_ratio": 0.45,
                    "anisotropy_index": 0.30,
                },
                # Layer 11, epsilon 0.001 — collapsed
                {
                    "layer": 11,
                    "epsilon": 0.001,
                    "effective_rank": 25.0,
                    "concentration_ratio": 0.22,
                    "anisotropy_index": 0.01,
                },
                # Layer 11, epsilon 0.01 — best
                {
                    "layer": 11,
                    "epsilon": 0.01,
                    "effective_rank": 75.0,
                    "concentration_ratio": 0.33,
                    "anisotropy_index": 0.06,
                },
                # Layer 11, epsilon 0.1 — too anisotropic
                {
                    "layer": 11,
                    "epsilon": 0.1,
                    "effective_rank": 92.0,
                    "concentration_ratio": 0.50,
                    "anisotropy_index": 0.55,
                },
            ],
        }

        with open(self.mock_results_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_data, f)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    # ------------------------------------------------------------------
    # 1. load_results
    # ------------------------------------------------------------------

    def test_load_results_parses_json(self):
        """load_results must correctly parse a valid JSON file."""
        data = load_results(self.mock_results_path)
        self.assertEqual(data["experiment"], "Test Experiment")
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 9)

    def test_load_results_file_not_found(self):
        """load_results must raise FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            load_results(os.path.join(self.temp_dir.name, "nonexistent.json"))

    # ------------------------------------------------------------------
    # 2. statistical_comparison
    # ------------------------------------------------------------------

    def test_statistical_comparison_structure(self):
        """statistical_comparison must return a dict keyed by epsilon strings."""
        comparisons = statistical_comparison(
            self.mock_data["results"], self.mock_data["baseline_ranks"]
        )
        self.assertIn("0.001", comparisons)
        self.assertIn("0.01", comparisons)
        self.assertIn("0.1", comparisons)

        for eps_str, comp in comparisons.items():
            self.assertIn("eff_rank_mean", comp)
            self.assertIn("eff_rank_std", comp)
            self.assertIn("eff_rank_vs_baseline", comp)
            self.assertIn("eff_rank_ratio", comp)
            self.assertIn("conc_ratio_mean", comp)
            self.assertIn("anisotropy_mean", comp)
            self.assertIn("num_layers", comp)

    def test_statistical_comparison_values(self):
        """statistical_comparison must compute correct means and stds."""
        comparisons = statistical_comparison(
            self.mock_data["results"], self.mock_data["baseline_ranks"]
        )
        # Epsilon 0.01 has 3 layers with concentration ratios 0.35, 0.30, 0.33
        self.assertAlmostEqual(
            comparisons["0.01"]["conc_ratio_mean"], 0.326666, places=5
        )
        # Baseline avg = (100 + 90 + 80) / 3 = 90; eff_rank avg = (85 + 80 + 75) / 3 = 80
        self.assertAlmostEqual(
            comparisons["0.01"]["eff_rank_vs_baseline"], -10.0, places=3
        )

    def test_statistical_comparison_empty_results(self):
        """statistical_comparison must handle empty results gracefully."""
        comparisons = statistical_comparison({}, {})
        self.assertEqual(comparisons, {})

    def test_statistical_comparison_missing_baseline(self):
        """statistical_comparison must use default baseline of 100 when missing."""
        results = [
            {
                "layer": 3,
                "epsilon": 0.01,
                "effective_rank": 50.0,
                "concentration_ratio": 0.4,
            }
        ]
        comparisons = statistical_comparison(results, {})
        self.assertAlmostEqual(comparisons["0.01"]["eff_rank_vs_baseline"], -50.0)

    # ------------------------------------------------------------------
    # 3. layer_recommendations
    # ------------------------------------------------------------------

    def test_layer_recommendations_selects_best_epsilon(self):
        """layer_recommendations must pick the epsilon with lowest concentration ratio
        that satisfies the constraints (effective_rank >= 50% baseline, anisotropy < 0.5)."""
        recs = layer_recommendations(
            self.mock_data["results"], self.mock_data["baseline_ranks"]
        )
        # Layer 3: 0.001 is collapsed (30 < 50), 0.01 is good (cr=0.35), 0.1 is too anisotropic (0.6 > 0.5)
        self.assertEqual(recs[3]["recommended_epsilon"], 0.01)
        # Layer 7: 0.001 collapsed (20 < 45), 0.01 best (cr=0.30), 0.1 worse (cr=0.45)
        self.assertEqual(recs[7]["recommended_epsilon"], 0.01)
        # Layer 11: 0.001 collapsed (25 < 40), 0.01 best (cr=0.33), 0.1 too anisotropic (0.55 > 0.5)
        self.assertEqual(recs[11]["recommended_epsilon"], 0.01)

    def test_layer_recommendations_no_valid_epsilon(self):
        """If no epsilon satisfies constraints, recommended_epsilon should be None."""
        # All results collapsed (effective_rank far below 50% baseline)
        bad_results = [
            {
                "layer": 3,
                "epsilon": 0.01,
                "effective_rank": 1.0,
                "concentration_ratio": 0.1,
                "anisotropy_index": 0.01,
            }
        ]
        recs = layer_recommendations(bad_results, {"3": 100.0})
        self.assertIsNone(recs[3]["recommended_epsilon"])
        self.assertEqual(recs[3]["score"], float("inf"))

    def test_layer_recommendations_empty(self):
        """layer_recommendations must return an empty dict for empty results."""
        recs = layer_recommendations([], {})
        self.assertEqual(recs, {})

    # ------------------------------------------------------------------
    # 4. generate_report
    # ------------------------------------------------------------------

    def test_generate_report_contains_sections(self):
        """The generated report must contain expected markdown sections."""
        report = generate_report(self.mock_data)
        self.assertIn("# Post-Experiment Analysis Report", report)
        self.assertIn("## 1. Sweet Spot Summary", report)
        self.assertIn("## 2. Statistical Comparison vs Baseline", report)
        self.assertIn("## 3. Per-Layer Recommendations", report)
        self.assertIn("## 4. Key Findings", report)
        self.assertIn("## 5. Recommendations for Next Steps", report)

    def test_generate_report_includes_epsilon_values(self):
        """The report table must list all epsilon values tested."""
        report = generate_report(self.mock_data)
        for eps in self.mock_data["config"]["epsilon_values"]:
            self.assertIn(str(eps), report)

    def test_generate_report_empty_results(self):
        """generate_report must not crash when results list is empty."""
        data = {
            "experiment": "Empty",
            "date": "2026-05-09",
            "config": {},
            "baseline_ranks": {},
            "sweet_spot": {},
            "results": [],
        }
        report = generate_report(data)
        self.assertIn("Post-Experiment Analysis Report", report)

    # ------------------------------------------------------------------
    # 5. export_optimal_config
    # ------------------------------------------------------------------

    def test_export_optimal_config_is_valid_python(self):
        """export_optimal_config must return a string that can be compiled as Python."""
        sweet_spot = self.mock_data["sweet_spot"]
        recs = layer_recommendations(
            self.mock_data["results"], self.mock_data["baseline_ranks"]
        )
        config_code = export_optimal_config(sweet_spot, recs)
        # Verify it compiles without syntax errors
        compile(config_code, "<string>", "exec")
        self.assertIn("OPTIMAL_EPSILON", config_code)
        self.assertIn("LAYER_EPSILON_MAP", config_code)

    def test_export_optimal_config_with_none_recommendations(self):
        """export_optimal_config must handle None recommendations gracefully."""
        sweet_spot = {"epsilon": 0.01, "layers": [3]}
        recs = {
            3: {
                "recommended_epsilon": None,
                "baseline_rank": 100.0,
                "score": float("inf"),
            }
        }
        config_code = export_optimal_config(sweet_spot, recs)
        compile(config_code, "<string>", "exec")
        self.assertIn("null", config_code)

    # ------------------------------------------------------------------
    # 6. Edge cases / integration
    # ------------------------------------------------------------------

    def test_full_pipeline_with_mock_file(self):
        """End-to-end: load mock file, generate report, export config."""
        data = load_results(self.mock_results_path)
        report = generate_report(data)
        recs = layer_recommendations(data["results"], data["baseline_ranks"])
        config_code = export_optimal_config(data["sweet_spot"], recs)
        self.assertGreater(len(report), 0)
        self.assertGreater(len(config_code), 0)

    def test_missing_keys_in_results(self):
        """Functions must not crash when result dicts lack optional keys."""
        sparse_results = [
            {
                "layer": 3,
                "epsilon": 0.01,
            },  # missing concentration_ratio, anisotropy_index, effective_rank
        ]
        comparisons = statistical_comparison(sparse_results, {})
        self.assertIn("0.01", comparisons)
        self.assertEqual(comparisons["0.01"]["conc_ratio_mean"], 1.0)  # default value
        self.assertEqual(comparisons["0.01"]["anisotropy_mean"], 1.0)  # default value

    def test_report_with_missing_config_keys(self):
        """generate_report must handle missing config keys gracefully."""
        data = {
            "experiment": "Minimal",
            "date": "2026-05-09",
            "config": {},
            "baseline_ranks": {},
            "sweet_spot": {},
            "results": [],
        }
        report = generate_report(data)
        # When keys are missing, the report uses None or empty defaults
        self.assertIn("None", report)
        self.assertIn("Post-Experiment Analysis Report", report)


if __name__ == "__main__":
    unittest.main()
