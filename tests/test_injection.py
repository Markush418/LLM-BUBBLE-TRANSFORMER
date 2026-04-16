"""
Tests for SDOT Injection Script
================================

Unit tests for:
- inject_sdot_into_model: Model injection
- create_mock_qwen_model: Mock model creation
- load_calibration_thresholds: Threshold loading
"""

import sys
import os
import unittest
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.inject_sdot_qwen import (
    inject_sdot_into_model,
    create_mock_qwen_model,
    load_calibration_thresholds,
    create_sdot_attention_layer,
)
from models.sdot_attention import SDOTAttention


class TestLoadCalibrationThresholds(unittest.TestCase):
    """Tests for load_calibration_thresholds function."""

    def test_load_existing_thresholds(self):
        """Should load thresholds from existing file."""
        # Create temporary thresholds file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            thresholds = {
                "layer_3": {"optimal_C": 32, "hard_support": 1024},
                "layer_7": {"optimal_C": 64, "hard_support": 2048},
            }
            json.dump(thresholds, f)
            temp_path = f.name

        try:
            result = load_calibration_thresholds(temp_path)

            # Check structure
            self.assertIn("layer_3", result)
            self.assertIn("layer_7", result)

            # Check values
            self.assertEqual(result["layer_3"]["optimal_C"], 32)
            self.assertEqual(result["layer_7"]["optimal_C"], 64)

        finally:
            os.unlink(temp_path)

    def test_load_missing_thresholds(self):
        """Should return empty dict for missing file."""
        result = load_calibration_thresholds("nonexistent_path.json")

        # Should return empty dict
        self.assertEqual(result, {})


class TestCreateSDOTAttentionLayer(unittest.TestCase):
    """Tests for create_sdot_attention_layer function."""

    def test_create_layer_with_fixed_C(self):
        """Should create layer with fixed C."""
        layer = create_sdot_attention_layer(
            d_model=896,
            num_heads=14,
            optimal_C=32,
            use_baroreceptor=False,
        )

        # Check type
        self.assertIsInstance(layer, SDOTAttention)

        # Check config
        self.assertEqual(layer.d_model, 896)
        self.assertEqual(layer.num_heads, 14)
        self.assertEqual(layer.num_centroids, 32)
        self.assertFalse(layer.use_baroreceptor)

    def test_create_layer_with_baroreceptor(self):
        """Should create layer with baroreceptor."""
        layer = create_sdot_attention_layer(
            d_model=512,
            num_heads=8,
            optimal_C=64,
            use_baroreceptor=True,
        )

        # Check type
        self.assertIsInstance(layer, SDOTAttention)

        # Check baroreceptor enabled
        self.assertTrue(layer.use_baroreceptor)


class TestCreateMockQwenModel(unittest.TestCase):
    """Tests for create_mock_qwen_model function."""

    def test_mock_model_structure(self):
        """Should create model with correct structure."""
        model = create_mock_qwen_model(d_model=896, num_layers=24, num_heads=14)

        # Check structure
        self.assertTrue(hasattr(model, "model"))
        self.assertTrue(hasattr(model.model, "layers"))
        self.assertEqual(len(model.model.layers), 24)

    def test_mock_model_forward(self):
        """Should run forward pass."""
        model = create_mock_qwen_model(d_model=512, num_layers=6, num_heads=8)

        # Create input
        x = torch.randn(2, 64, 512)

        # Forward pass
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (2, 64, 512))


class TestInjectSDOTIntoModel(unittest.TestCase):
    """Tests for inject_sdot_into_model function."""

    def test_inject_single_layer(self):
        """Should inject SDOT into single layer."""
        model = create_mock_qwen_model(d_model=512, num_layers=6, num_heads=8)

        thresholds = {"layer_3": {"optimal_C": 32, "hard_support": 1024}}

        injected = inject_sdot_into_model(
            model=model,
            target_layers=[3],
            thresholds=thresholds,
            d_model=512,
            num_heads=8,
        )

        # Check that layer 3 was replaced
        self.assertIsInstance(injected.model.layers[3].self_attn, SDOTAttention)

    def test_inject_multiple_layers(self):
        """Should inject SDOT into multiple layers."""
        model = create_mock_qwen_model(d_model=512, num_layers=12, num_heads=8)

        thresholds = {
            "layer_3": {"optimal_C": 32, "hard_support": 1024},
            "layer_7": {"optimal_C": 64, "hard_support": 2048},
        }

        injected = inject_sdot_into_model(
            model=model,
            target_layers=[3, 7],
            thresholds=thresholds,
            d_model=512,
            num_heads=8,
        )

        # Check that both layers were replaced
        self.assertIsInstance(injected.model.layers[3].self_attn, SDOTAttention)
        self.assertIsInstance(injected.model.layers[7].self_attn, SDOTAttention)

    def test_inject_with_default_C(self):
        """Should use default C when not in thresholds."""
        model = create_mock_qwen_model(d_model=512, num_layers=6, num_heads=8)

        # Empty thresholds
        thresholds = {}

        injected = inject_sdot_into_model(
            model=model,
            target_layers=[3],
            thresholds=thresholds,
            d_model=512,
            num_heads=8,
            default_C=48,
        )

        # Check that layer was replaced with default C
        self.assertIsInstance(injected.model.layers[3].self_attn, SDOTAttention)
        self.assertEqual(injected.model.layers[3].self_attn.num_centroids, 48)

    def test_injected_model_forward(self):
        """Should run forward pass after injection."""
        model = create_mock_qwen_model(d_model=512, num_layers=6, num_heads=8)

        thresholds = {"layer_3": {"optimal_C": 32, "hard_support": 1024}}

        injected = inject_sdot_into_model(
            model=model,
            target_layers=[3],
            thresholds=thresholds,
            d_model=512,
            num_heads=8,
        )

        # Create input
        x = torch.randn(2, 64, 512)

        # Forward pass
        output = injected(x)

        # Check output shape
        self.assertEqual(output.shape, (2, 64, 512))

    def test_inject_preserves_other_layers(self):
        """Should not modify non-target layers."""
        model = create_mock_qwen_model(d_model=512, num_layers=6, num_heads=8)

        # Store original layer 0
        original_layer0 = model.model.layers[0].self_attn

        thresholds = {"layer_3": {"optimal_C": 32, "hard_support": 1024}}

        injected = inject_sdot_into_model(
            model=model,
            target_layers=[3],
            thresholds=thresholds,
            d_model=512,
            num_heads=8,
        )

        # Layer 0 should be unchanged
        self.assertIs(injected.model.layers[0].self_attn, original_layer0)


class TestIntegration(unittest.TestCase):
    """Integration tests for injection pipeline."""

    def test_full_injection_pipeline(self):
        """Test full injection pipeline."""
        # Create mock model
        model = create_mock_qwen_model(d_model=896, num_layers=24, num_heads=14)

        # Create thresholds
        thresholds = {
            "layer_3": {"optimal_C": 32, "hard_support": 1024},
            "layer_7": {"optimal_C": 64, "hard_support": 2048},
            "layer_11": {"optimal_C": 128, "hard_support": 4096},
        }

        # Inject SDOT
        injected = inject_sdot_into_model(
            model=model,
            target_layers=[3, 7, 11],
            thresholds=thresholds,
            d_model=896,
            num_heads=14,
        )

        # Verify injection
        for layer_idx in [3, 7, 11]:
            self.assertIsInstance(
                injected.model.layers[layer_idx].self_attn, SDOTAttention
            )

        # Test forward pass
        x = torch.randn(2, 128, 896)
        output = injected(x)

        # Verify output
        self.assertEqual(output.shape, (2, 128, 896))
        self.assertFalse(torch.isnan(output).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
