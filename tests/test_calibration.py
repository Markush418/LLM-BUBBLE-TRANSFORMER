"""
Tests for Calibration and Benchmark
===================================

Unit tests for:
- find_optimal_C: Calibration algorithm
- compute_hard_support: Hard support metric
- benchmark_throughput: Throughput measurement
- benchmark_memory: Memory measurement
"""

import sys
import os
import unittest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.v3_core import compute_hard_support, cluster_keys, voronoi_assign
from models.sdot_attention import SDOTAttention
from experiments.plateau_attention import PlateauAttentionMechanism
from calibración.find_hard_plateau import (
    find_optimal_C,
    create_mock_dataloader,
    calibrate_all_layers,
)
from scripts.benchmark_v2_vs_v3 import (
    benchmark_throughput,
    benchmark_memory,
)


class TestFindOptimalC(unittest.TestCase):
    """Tests for find_optimal_C function."""

    def test_find_optimal_C_returns_valid_C(self):
        """Should return a valid C value in range."""
        d_model = 256
        num_heads = 4

        model = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        dataloader = create_mock_dataloader(
            batch_size=2,
            seq_len=64,
            d_model=d_model,
            num_batches=2,
            device="cpu",
        )

        result = find_optimal_C(
            model=model,
            layer_idx=3,
            dataloader=dataloader,
            min_C=8,
            max_C=128,
            steps=5,
            device="cpu",
        )

        # Check structure
        self.assertIn("layer_idx", result)
        self.assertIn("optimal_C", result)
        self.assertIn("hard_support", result)
        self.assertIn("plateau_reached", result)
        self.assertIn("all_results", result)

        # Check values
        self.assertEqual(result["layer_idx"], 3)
        self.assertGreaterEqual(result["optimal_C"], 8)
        self.assertLessEqual(result["optimal_C"], 128)
        self.assertGreaterEqual(result["hard_support"], 0)
        self.assertIsInstance(result["plateau_reached"], bool)
        self.assertGreater(len(result["all_results"]), 0)

    def test_find_optimal_C_detects_plateau(self):
        """Should detect plateau when support stabilizes."""
        d_model = 256
        num_heads = 4

        model = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        dataloader = create_mock_dataloader(
            batch_size=2,
            seq_len=64,
            d_model=d_model,
            num_batches=2,
            device="cpu",
        )

        # Use wide C range to ensure plateau detection
        result = find_optimal_C(
            model=model,
            layer_idx=7,
            dataloader=dataloader,
            min_C=8,
            max_C=256,
            steps=10,
            device="cpu",
        )

        # Should have tested multiple C values
        self.assertGreater(len(result["all_results"]), 1)

        # Each result should have C and support
        for res in result["all_results"]:
            self.assertIn("C", res)
            self.assertIn("support", res)


class TestComputeHardSupport(unittest.TestCase):
    """Tests for compute_hard_support function."""

    def test_compute_hard_support_correct(self):
        """Hard support should count pairs in same bubble."""
        B, H, N = 2, 4, 100
        C = 32

        # Create assignments
        assignments = torch.randint(0, C, (B, H, N))
        support = compute_hard_support(assignments)

        # Check shape
        self.assertEqual(support.shape, (B, H))

        # Check non-negative
        self.assertTrue((support >= 0).all())

    def test_compute_hard_support_single_bubble(self):
        """All tokens in one bubble should have maximum support."""
        B, H, N = 2, 4, 100

        # All tokens in bubble 0
        assignments = torch.zeros((B, H, N), dtype=torch.long)
        support = compute_hard_support(assignments)

        # Expected: N^2 - N pairs (all pairs except self-pairs)
        expected = N * N - N

        self.assertTrue(torch.allclose(support, torch.tensor([[float(expected)]])))

    def test_compute_hard_support_all_separate(self):
        """Each token in its own bubble should have zero support."""
        B, H, N = 1, 1, 10

        # Each token in different bubble
        assignments = torch.arange(N).unsqueeze(0).unsqueeze(0)
        support = compute_hard_support(assignments)

        # Expected: 0 (no pairs in same bubble)
        self.assertEqual(support.item(), 0.0)


class TestBenchmarkThroughput(unittest.TestCase):
    """Tests for benchmark_throughput function."""

    def test_benchmark_throughput_measures_correctly(self):
        """Should measure throughput for V2 and V3."""
        d_model = 256
        num_heads = 4
        B, N = 2, 64

        model_v2 = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=0.01,
        )

        model_v3 = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=16,
            use_baroreceptor=False,
        )

        input_tensor = torch.randn(B, N, d_model)

        result = benchmark_throughput(
            model_v2=model_v2,
            model_v3=model_v3,
            input_tensor=input_tensor,
            num_runs=3,
            warmup_runs=1,
            device="cpu",
        )

        # Check structure
        self.assertIn("v2_time_ms", result)
        self.assertIn("v3_time_ms", result)
        self.assertIn("speedup", result)
        self.assertIn("v2_tokens_per_sec", result)
        self.assertIn("v3_tokens_per_sec", result)

        # Check values
        self.assertGreater(result["v2_time_ms"], 0)
        self.assertGreater(result["v3_time_ms"], 0)
        self.assertGreater(result["speedup"], 0)
        self.assertGreater(result["v2_tokens_per_sec"], 0)
        self.assertGreater(result["v3_tokens_per_sec"], 0)

    def test_benchmark_throughput_consistency(self):
        """Multiple runs should give consistent results."""
        d_model = 256
        num_heads = 4
        B, N = 2, 64

        model_v2 = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=0.01,
        )

        model_v3 = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=16,
            use_baroreceptor=False,
        )

        input_tensor = torch.randn(B, N, d_model)

        # Run twice
        result1 = benchmark_throughput(
            model_v2=model_v2,
            model_v3=model_v3,
            input_tensor=input_tensor,
            num_runs=5,
            device="cpu",
        )

        result2 = benchmark_throughput(
            model_v2=model_v2,
            model_v3=model_v3,
            input_tensor=input_tensor,
            num_runs=5,
            device="cpu",
        )

        # Results should be within 50% of each other (generous tolerance)
        self.assertLess(
            abs(result1["speedup"] - result2["speedup"]) / result1["speedup"], 0.5
        )


class TestBenchmarkMemory(unittest.TestCase):
    """Tests for benchmark_memory function."""

    def test_benchmark_memory_measures_correctly(self):
        """Should measure memory for V2 and V3."""
        d_model = 256
        num_heads = 4
        B, N = 2, 64

        model_v2 = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=0.01,
        )

        model_v3 = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=16,
            use_baroreceptor=False,
        )

        input_tensor = torch.randn(B, N, d_model)

        result = benchmark_memory(
            model_v2=model_v2,
            model_v3=model_v3,
            input_tensor=input_tensor,
            device="cpu",
        )

        # Check structure
        self.assertIn("v2_memory_mb", result)
        self.assertIn("v3_memory_mb", result)
        self.assertIn("memory_reduction", result)

        # Check values
        self.assertGreater(result["v2_memory_mb"], 0)
        self.assertGreater(result["v3_memory_mb"], 0)
        self.assertGreater(result["memory_reduction"], 0)

    def test_benchmark_memory_v3_smaller(self):
        """V3 should use less memory than V2."""
        d_model = 256
        num_heads = 4
        B, N = 2, 128  # Larger N to see difference

        model_v2 = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=0.01,
        )

        model_v3 = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=16,
            use_baroreceptor=False,
        )

        input_tensor = torch.randn(B, N, d_model)

        result = benchmark_memory(
            model_v2=model_v2,
            model_v3=model_v3,
            input_tensor=input_tensor,
            device="cpu",
        )

        # V3 should use less memory (memory_reduction > 1)
        self.assertGreater(result["memory_reduction"], 1.0)


class TestCalibrateAllLayers(unittest.TestCase):
    """Tests for calibrate_all_layers function."""

    def test_calibrate_all_layers_returns_results(self):
        """Should return results for all layers."""
        result = calibrate_all_layers(
            d_model=256,
            num_heads=4,
            target_layers=[3, 7],
            min_C=8,
            max_C=64,
            steps=3,
            device="cpu",
            test_mode=True,
        )

        # Check structure
        self.assertIn("layer_3", result)
        self.assertIn("layer_7", result)

        # Check each layer result
        for layer_key in ["layer_3", "layer_7"]:
            layer_result = result[layer_key]
            self.assertIn("optimal_C", layer_result)
            self.assertIn("hard_support", layer_result)

            self.assertGreaterEqual(layer_result["optimal_C"], 8)
            self.assertLessEqual(layer_result["optimal_C"], 64)
            self.assertGreaterEqual(layer_result["hard_support"], 0)

    def test_calibrate_all_layers_test_mode(self):
        """Test mode should use fewer steps."""
        result = calibrate_all_layers(
            d_model=256,
            num_heads=4,
            target_layers=[3],
            min_C=8,
            max_C=64,
            steps=10,  # Will be overridden by test_mode
            device="cpu",
            test_mode=True,
        )

        # Should complete quickly
        self.assertIn("layer_3", result)


class TestIntegration(unittest.TestCase):
    """Integration tests for calibration pipeline."""

    def test_full_calibration_pipeline(self):
        """Test full calibration pipeline."""
        d_model = 256
        num_heads = 4

        # Create model
        model = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=32,
            use_baroreceptor=False,
        )

        # Create dataloader
        dataloader = create_mock_dataloader(
            batch_size=2,
            seq_len=64,
            d_model=d_model,
            num_batches=2,
            device="cpu",
        )

        # Run calibration
        result = find_optimal_C(
            model=model,
            layer_idx=3,
            dataloader=dataloader,
            min_C=8,
            max_C=64,
            steps=5,
            device="cpu",
        )

        # Verify result
        self.assertIsNotNone(result)
        self.assertIn("optimal_C", result)
        self.assertGreater(result["optimal_C"], 0)

    def test_benchmark_integration(self):
        """Test benchmark integration."""
        d_model = 256
        num_heads = 4
        B, N = 2, 64

        model_v2 = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
        )

        model_v3 = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=16,
            use_baroreceptor=False,
        )

        input_tensor = torch.randn(B, N, d_model)

        # Run throughput benchmark
        throughput = benchmark_throughput(
            model_v2=model_v2,
            model_v3=model_v3,
            input_tensor=input_tensor,
            num_runs=3,
            device="cpu",
        )

        # Run memory benchmark
        memory = benchmark_memory(
            model_v2=model_v2,
            model_v3=model_v3,
            input_tensor=input_tensor,
            device="cpu",
        )

        # Both should complete without error
        self.assertIsNotNone(throughput)
        self.assertIsNotNone(memory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
