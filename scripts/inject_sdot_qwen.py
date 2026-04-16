"""
Integration Script: Inject SDOT into Qwen3-0.6B
================================================

Replaces standard attention with SDOT attention in target layers.

Target layers (Qwen3-0.6B full-attention): [3, 7, 11, 15, 19, 23]
Config: d_model=896, num_heads=14, head_dim=64

Usage:
    python scripts/inject_sdot_qwen.py --output models/qwen3-sdot-v3
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.sdot_attention import SDOTAttention


def load_calibration_thresholds(
    thresholds_path: str = "calibración/layer_thresholds.json",
) -> Dict[str, Dict]:
    """
    Load calibrated C values for each layer.

    Args:
        thresholds_path: Path to layer_thresholds.json

    Returns:
        Dict mapping layer_idx to optimal_C and hard_support

    Example:
        {
            "layer_3": {"optimal_C": 32, "hard_support": 1024},
            "layer_7": {"optimal_C": 64, "hard_support": 2048},
            ...
        }
    """
    if not os.path.exists(thresholds_path):
        print(f"[Warning] Thresholds file not found: {thresholds_path}")
        print("[Warning] Using default C=32 for all layers")
        return {}

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    print(f"[Info] Loaded calibration thresholds from {thresholds_path}")
    return thresholds


def create_sdot_attention_layer(
    d_model: int = 896,
    num_heads: int = 14,
    optimal_C: int = 32,
    use_baroreceptor: bool = False,
) -> SDOTAttention:
    """
    Create SDOTAttention layer with calibrated C.

    Args:
        d_model: Model dimension (Qwen3-0.6B: 896)
        num_heads: Number of attention heads (Qwen3-0.6B: 14)
        optimal_C: Number of centroids (from calibration)
        use_baroreceptor: If True, use dynamic C prediction

    Returns:
        SDOTAttention module ready for injection
    """
    sdot_layer = SDOTAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_centroids=optimal_C,
        use_baroreceptor=use_baroreceptor,
    )

    return sdot_layer


def inject_sdot_into_model(
    model: nn.Module,
    target_layers: list = None,
    thresholds: Dict[str, Dict] = None,
    d_model: int = 896,
    num_heads: int = 14,
    default_C: int = 32,
    use_baroreceptor: bool = False,
) -> nn.Module:
    """
    Inject SDOT attention into model's target layers.

    Args:
        model: Transformer model (e.g., Qwen3-0.6B)
        target_layers: List of layer indices to replace
        thresholds: Calibration thresholds dict
        d_model: Model dimension
        num_heads: Number of attention heads
        default_C: Default C if not in thresholds
        use_baroreceptor: If True, use dynamic C

    Returns:
        Model with SDOT attention in target layers
    """
    if target_layers is None:
        # Qwen3-0.6B full-attention layers
        target_layers = [3, 7, 11, 15, 19, 23]

    if thresholds is None:
        thresholds = {}

    print(f"\n[Injection] Target layers: {target_layers}")

    injected_count = 0

    for layer_idx in target_layers:
        # Get optimal C for this layer
        layer_key = f"layer_{layer_idx}"

        if layer_key in thresholds:
            optimal_C = thresholds[layer_key]["optimal_C"]
            print(f"  Layer {layer_idx}: Using calibrated C={optimal_C}")
        else:
            optimal_C = default_C
            print(f"  Layer {layer_idx}: Using default C={optimal_C}")

        # Create SDOT attention
        sdot_attn = create_sdot_attention_layer(
            d_model=d_model,
            num_heads=num_heads,
            optimal_C=optimal_C,
            use_baroreceptor=use_baroreceptor,
        )

        # Inject into model
        # Note: This assumes model structure like model.model.layers[i].self_attn
        # Adjust based on actual Qwen3 architecture
        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                model.model.layers[layer_idx].self_attn = sdot_attn
                injected_count += 1
            else:
                print(
                    f"  [Warning] Model structure not recognized, skipping layer {layer_idx}"
                )
        except (AttributeError, IndexError) as e:
            print(f"  [Warning] Failed to inject layer {layer_idx}: {e}")

    print(
        f"\n[Injection] Successfully injected {injected_count}/{len(target_layers)} layers"
    )

    return model


def create_mock_qwen_model(
    d_model: int = 896,
    num_layers: int = 24,
    num_heads: int = 14,
):
    """
    Create a mock Qwen3-0.6B model for testing.

    Returns:
        Mock model with similar structure to Qwen3
    """

    class MockAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, x):
            B, N, D = x.shape
            Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, V)
            out = out.transpose(1, 2).reshape(B, N, D)
            return self.W_o(out)

    class MockLayer(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.self_attn = MockAttention(d_model, num_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )

        def forward(self, x):
            # Handle both MockAttention (returns tensor) and SDOTAttention (returns tuple)
            attn_out = self.self_attn(self.norm1(x))
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]  # Extract output from (output, assignments)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [MockLayer(d_model, num_heads) for _ in range(num_layers)]
            )

        def forward(self, x):
            for layer in self.model.layers:
                x = layer(x)
            return x

    return MockModel()


def test_injection():
    """Test injection with mock model."""
    print("\n[TEST] Testing injection with mock model...")

    # Create mock model
    mock_model = create_mock_qwen_model(d_model=896, num_layers=24, num_heads=14)

    # Create mock thresholds
    mock_thresholds = {
        "layer_3": {"optimal_C": 32, "hard_support": 1024},
        "layer_7": {"optimal_C": 64, "hard_support": 2048},
        "layer_11": {"optimal_C": 128, "hard_support": 4096},
    }

    # Inject SDOT
    injected_model = inject_sdot_into_model(
        model=mock_model,
        target_layers=[3, 7, 11],
        thresholds=mock_thresholds,
        d_model=896,
        num_heads=14,
    )

    # Test forward pass
    test_input = torch.randn(2, 128, 896)
    output = injected_model(test_input)

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("[TEST] Injection test passed!")

    return injected_model


def main():
    parser = argparse.ArgumentParser(description="Inject SDOT into Qwen3-0.6B")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="calibración/layer_thresholds.json",
        help="Path to calibration thresholds JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/qwen3-sdot-v3",
        help="Output directory for injected model",
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=[3, 7, 11, 15, 19, 23],
        help="Target layer indices",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=896,
        help="Model dimension (Qwen3-0.6B: 896)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=14,
        help="Number of attention heads (Qwen3-0.6B: 14)",
    )
    parser.add_argument(
        "--default-C",
        type=int,
        default=32,
        help="Default C if not in thresholds",
    )
    parser.add_argument(
        "--use-baroreceptor",
        action="store_true",
        help="Use dynamic C prediction",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test with mock model (no real Qwen3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SDOT Injection into Qwen3-0.6B")
    print("=" * 60)

    # Load thresholds
    thresholds = load_calibration_thresholds(args.thresholds)

    if args.test_mode:
        # Test mode: use mock model
        print("\n[Mode] Test mode with mock model")
        model = create_mock_qwen_model(
            d_model=args.d_model,
            num_layers=24,
            num_heads=args.num_heads,
        )
    else:
        # Real mode: load Qwen3
        print(f"\n[Mode] Loading real model: {args.model_name}")
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            print(f"[Info] Model loaded successfully")
        except ImportError:
            print("[Error] transformers library not installed")
            print("[Error] Run: pip install transformers")
            print("[Info] Falling back to test mode")
            model = create_mock_qwen_model(
                d_model=args.d_model,
                num_layers=24,
                num_heads=args.num_heads,
            )

    # Inject SDOT
    model = inject_sdot_into_model(
        model=model,
        target_layers=args.target_layers,
        thresholds=thresholds,
        d_model=args.d_model,
        num_heads=args.num_heads,
        default_C=args.default_C,
        use_baroreceptor=args.use_baroreceptor,
    )

    # Save model
    if not args.test_mode:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            model.save_pretrained(str(output_dir))
            print(f"\n[Success] Model saved to {output_dir}")
        except AttributeError:
            print(f"\n[Warning] Model doesn't support save_pretrained, skipping save")

    print("\n" + "=" * 60)
    print("Injection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
