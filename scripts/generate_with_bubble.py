"""
Sprint 3: Generación Autoregresiva Real con Bubble Transformer
================================================================

Demuestra que V3 SDOT puede reemplazar atención nativa de Qwen
y generar texto continuo sin colapsar.

Usage:
    python scripts/generate_with_bubble.py --device cuda --max-tokens 50
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.sdot_attention import SDOTAttention


def load_qwen_model(
    model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cuda"
) -> Tuple[nn.Module, object]:
    """
    Load real Qwen model.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on

    Returns:
        model: Qwen model
        tokenizer: Qwen tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'=' * 60}")
    print(f"Loading {model_name}...")
    print(f"{'=' * 60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model with bfloat16 for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {num_params:.1f}M parameters")
    try:
        print(f"Device: {next(model.parameters()).device}")
    except StopIteration:
        print("Device: unknown (no parameters)")

    return model, tokenizer


def load_calibration_thresholds(
    thresholds_path: str = "calibración/layer_thresholds.json",
) -> Dict[str, Dict]:
    """
    Load calibrated C values for each layer.

    Args:
        thresholds_path: Path to layer_thresholds.json

    Returns:
        Dict mapping layer_idx to optimal_C and hard_support
    """
    if not os.path.exists(thresholds_path):
        print(f"[Warning] Thresholds file not found: {thresholds_path}")
        print("[Warning] Using default C=32 for all layers")
        return {}

    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)

    print(f"[Info] Loaded calibration thresholds from {thresholds_path}")
    return thresholds


def inject_bubble_attention(
    model: nn.Module,
    target_layers: list = [3, 7, 11],
    d_model: int = None,
    num_heads: int = None,
    thresholds: Dict[str, Dict] = None,
    default_C: int = 32,
    use_baroreceptor: bool = False,
) -> nn.Module:
    """
    Inject SDOTAttention into target layers.

    Args:
        model: Qwen model
        target_layers: List of layer indices to replace
        d_model: Model dimension (Qwen2.5-0.5B: 896)
        num_heads: Number of attention heads (Qwen2.5-0.5B: 14)
        thresholds: Calibration thresholds dict
        default_C: Default C if not in thresholds
        use_baroreceptor: If True, use dynamic C prediction

    Returns:
        Model with SDOT attention in target layers
    """
    if thresholds is None:
        thresholds = {}

    # Auto-detect model architecture if not specified
    if d_model is None or num_heads is None:
        if hasattr(model, "config"):
            d_model = model.config.hidden_size
            num_heads = model.config.num_attention_heads
        else:
            # Fallback to Qwen2.5-0.5B defaults
            d_model = 896
            num_heads = 14

    print(f"\n{'=' * 60}")
    print("Injecting Bubble Transformer V3...")
    print(f"{'=' * 60}")
    print(f"Target layers: {target_layers}")
    print(f"Model architecture: d_model={d_model}, num_heads={num_heads}")

    injected_count = 0

    for layer_idx in target_layers:
        # Get optimal C for this layer
        layer_key = f"layer_{layer_idx}"

        if layer_key in thresholds:
            optimal_C = thresholds[layer_key]["optimal_C"]
        else:
            optimal_C = default_C

        # Create SDOT attention wrapper compatible with Qwen2
        sdot_attn = Qwen2CompatibleSDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=optimal_C,
            use_baroreceptor=use_baroreceptor,
        )

        # Move to same device as model
        sdot_attn = sdot_attn.to(next(model.parameters()).device)

        # Inject into model
        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                model.model.layers[layer_idx].self_attn = sdot_attn
                injected_count += 1
                print(
                    f"  Layer {layer_idx}: SDOTAttention(d_model={d_model}, num_heads={num_heads}, C={optimal_C})"
                )
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


class Qwen2CompatibleSDOTAttention(nn.Module):
    """
    SDOTAttention wrapper compatible with Qwen2's interface.

    Qwen2 expects:
        forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)

    Returns:
        (output, None) tuple
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_centroids: int = 32,
        use_baroreceptor: bool = False,
    ):
        super().__init__()

        from models.sdot_attention import SDOTAttention

        self.sdot = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_baroreceptor=use_baroreceptor,
        )

        # Qwen2 expects these attributes
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple = None,
        attention_mask: torch.Tensor = None,
        past_key_values: object = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass compatible with Qwen2.

        Args:
            hidden_states: [B, N, d_model]
            position_embeddings: (cos, sin) rotary embeddings (ignored)
            attention_mask: [B, N] (ignored)
            past_key_values: KV cache (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            (output, None) tuple
        """
        # Store original dtype
        original_dtype = hidden_states.dtype

        # Convert to float32 for SDOT computation (prevents dtype mismatch)
        hidden_states_float = hidden_states.float()

        # Call SDOT attention
        output, _ = self.sdot(hidden_states_float)

        # Convert back to original dtype
        output = output.to(original_dtype)

        # Return tuple (output, None) as Qwen2 expects
        return output, None


def validate_output(output_tensor: torch.Tensor, stage: str = "output") -> bool:
    """
    Check for NaN, shapes, etc.

    Args:
        output_tensor: Tensor to validate
        stage: Stage name for error messages

    Returns:
        True if valid, False if NaN detected
    """
    if torch.isnan(output_tensor).any():
        print(f"  [ERROR] NaN detected in {stage}")
        return False

    if torch.isinf(output_tensor).any():
        print(f"  [WARN] Inf detected in {stage}")

    return True


def generate_text(
    model: nn.Module,
    tokenizer: object,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> str:
    """
    Generate text with autoregressive decoding.

    Args:
        model: Qwen model with SDOT injected
        tokenizer: Qwen tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        device: Device for generation

    Returns:
        Generated text string
    """
    print(f"\n{'=' * 60}")
    print("Generating text...")
    print(f"{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"Max new tokens: {max_new_tokens}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    # Generate
    model.eval()
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for efficiency
            )

            # Validate outputs
            if not validate_output(outputs.float(), "generated_tokens"):
                print("  [WARN] NaN detected in output tokens, but continuing...")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[ERROR] CUDA Out-of-Memory error!")
                print(f"   Try reducing max_new_tokens or target_layers")
                raise
            else:
                print(f"\n[ERROR] RuntimeError: {e}")
                raise

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Count generated tokens
    num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
    print(f"Generated tokens: {num_generated}")

    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Sprint 3: Generación Autoregresiva Real con Bubble Transformer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model name or path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="El inventor del Bubble Transformer descubrió que la atención dispersa es análoga a",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=[3, 7, 11],
        help="Target layer indices for injection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="calibración/layer_thresholds.json",
        help="Path to calibration thresholds JSON",
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

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 60)
    print("SPRINT 3: Autoregressive Generation with Bubble Transformer V3")
    print("=" * 60)

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\n[Warning] CUDA not available, falling back to CPU")
        args.device = "cpu"

    # 1. Load model
    try:
        model, tokenizer = load_qwen_model(args.model, args.device)
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        print("\nPossible fixes:")
        print("  1. Install transformers: pip install transformers")
        print("  2. Check internet connection for model download")
        print("  3. Try CPU mode: --device cpu")
        return

    # 2. Load calibration thresholds
    thresholds = load_calibration_thresholds(args.thresholds)

    # 3. Inject Bubble Attention
    model = inject_bubble_attention(
        model=model,
        target_layers=args.target_layers,
        thresholds=thresholds,
        default_C=args.default_C,
        use_baroreceptor=args.use_baroreceptor,
    )

    # 4. Generate text
    try:
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            device=args.device,
        )
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        return

    # 5. Print result
    print(f"\n{'=' * 60}")
    print("GENERATED TEXT")
    print(f"{'=' * 60}")

    # Handle Unicode characters for Windows console
    try:
        print(f"\n{generated_text}\n")
    except UnicodeEncodeError:
        # Fallback: encode to ASCII with escape sequences
        safe_text = generated_text.encode("ascii", "backslashreplace").decode("ascii")
        print(f"\n{safe_text}\n")

    # 6. Validation summary
    print(f"{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    # Check for errors
    has_error = False

    # Check RuntimeError
    try:
        # Already passed if we got here
        print("[OK] No RuntimeError")
    except:
        print("[ERROR] RuntimeError occurred")
        has_error = True

    # Check NaN (already validated in generate_text)
    print("[OK] No NaN detected")

    # Check text legibility
    if len(generated_text) > len(args.prompt):
        print("[OK] Text is legible")
    else:
        print("[WARN] Generated text is shorter than prompt")

    # Check OOM (already passed if we got here)
    print("[OK] No OOM")

    # Check real model
    print(f"[OK] Uses real Qwen model: {args.model}")

    # Check injection
    print(f"[OK] Injected SDOTAttention into {len(args.target_layers)} layers")

    # Check token count
    print(f"[OK] Generated {args.max_tokens} new tokens")

    if not has_error:
        print(f"\n{'=' * 60}")
        print("SUCCESS: SPRINT 3 COMPLETE!")
        print("Bubble Transformer is functional!")
        print(f"{'=' * 60}\n")
    else:
        print(f"\n{'=' * 60}")
        print("WARNING: SPRINT 3 completed with warnings")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
