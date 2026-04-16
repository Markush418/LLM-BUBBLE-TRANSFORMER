"""
Evaluate Bubble Transformer Performance
========================================

Comprehensive evaluation suite measuring:
1. Perplexity on validation text
2. Generation latency (ms per token)
3. Memory efficiency (MB during generation)
4. Text quality (basic coherence check)

Usage:
python scripts/evaluate_bubble.py --model Qwen/Qwen2.5-0.5B --target-layers 3 7 11 --output results/evaluation_report.json
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.sdot_attention import SDOTAttention


def load_qwen_model(
    model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cuda"
) -> Tuple[nn.Module, object]:
    """Load Qwen model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'=' * 60}")
    print(f"Loading {model_name}...")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {num_params:.1f}M parameters")

    return model, tokenizer


def inject_bubble_attention(
    model: nn.Module,
    target_layers: List[int],
    d_model: int = None,
    num_heads: int = None,
    thresholds: Dict[str, Dict] = None,
    default_C: int = 32,
) -> nn.Module:
    """Inject SDOTAttention into target layers."""
    from scripts.generate_with_bubble import Qwen2CompatibleSDOTAttention

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
        layer_key = f"layer_{layer_idx}"

        if layer_key in thresholds:
            optimal_C = thresholds[layer_key]["optimal_C"]
        else:
            optimal_C = default_C

        sdot_attn = Qwen2CompatibleSDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=optimal_C,
            use_baroreceptor=False,
        )

        sdot_attn = sdot_attn.to(next(model.parameters()).device)

        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                model.model.layers[layer_idx].self_attn = sdot_attn
                injected_count += 1
                print(f"  Layer {layer_idx}: SDOTAttention(C={optimal_C})")
        except (AttributeError, IndexError) as e:
            print(f"  [Warning] Failed to inject layer {layer_idx}: {e}")

    print(
        f"\n[Injection] Successfully injected {injected_count}/{len(target_layers)} layers"
    )

    return model


def measure_perplexity(
    model: nn.Module,
    tokenizer: object,
    text: str,
    device: str = "cuda",
    max_length: int = 512,
) -> float:
    """
    Calculate perplexity on text.

    Perplexity = exp(average negative log-likelihood)

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Text to evaluate
        device: Device
        max_length: Maximum sequence length

    Returns:
        Perplexity score (lower is better)
    """
    print(f"\n{'=' * 60}")
    print("Measuring Perplexity...")
    print(f"{'=' * 60}")

    # Tokenize
    encodings = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )

    if device == "cuda":
        encodings = {k: v.cuda() for k, v in encodings.items()}
    else:
        encodings = {k: v.to(device) for k, v in encodings.items()}

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    print(f"Input tokens: {input_ids.shape[1]}")

    # Calculate perplexity
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            neg_log_likelihood = outputs.loss.item()
            perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()
        except Exception as e:
            print(f"[Warning] Perplexity calculation failed: {e}")
            perplexity = float("inf")

    print(f"Perplexity: {perplexity:.2f}")

    return perplexity


def measure_latency(
    model: nn.Module,
    tokenizer: object,
    prompt: str,
    num_tokens: int = 50,
    device: str = "cuda",
    warmup_runs: int = 2,
    measure_runs: int = 5,
) -> Dict[str, float]:
    """
    Measure generation latency.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        num_tokens: Number of tokens to generate
        device: Device
        warmup_runs: Number of warmup runs
        measure_runs: Number of measurement runs

    Returns:
        Dict with latency metrics (ms per token)
    """
    print(f"\n{'=' * 60}")
    print("Measuring Latency...")
    print(f"{'=' * 60}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Tokens to generate: {num_tokens}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    # Warmup
    print(f"Warmup runs: {warmup_runs}")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

    # Measure
    print(f"Measurement runs: {measure_runs}")
    latencies = []

    for i in range(measure_runs):
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # Calculate metrics
    avg_latency_ms = sum(latencies) / len(latencies)
    ms_per_token = avg_latency_ms / num_tokens

    print(f"Average latency: {avg_latency_ms:.2f} ms")
    print(f"Latency per token: {ms_per_token:.2f} ms/token")

    return {
        "avg_latency_ms": avg_latency_ms,
        "ms_per_token": ms_per_token,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
    }


def measure_memory_usage(
    model: nn.Module,
    tokenizer: object,
    prompt: str,
    num_tokens: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure peak memory during generation.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        num_tokens: Number of tokens to generate
        device: Device

    Returns:
        Dict with memory metrics (MB)
    """
    print(f"\n{'=' * 60}")
    print("Measuring Memory Usage...")
    print(f"{'=' * 60}")

    if device != "cuda":
        print("[Warning] Memory measurement only available on CUDA")
        return {
            "peak_memory_mb": 0,
            "model_memory_mb": 0,
        }

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Get model memory
    model_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Model memory: {model_memory_mb:.2f} MB")

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Get peak memory
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory: {peak_memory_mb:.2f} MB")

    return {
        "peak_memory_mb": peak_memory_mb,
        "model_memory_mb": model_memory_mb,
        "generation_overhead_mb": peak_memory_mb - model_memory_mb,
    }


def measure_text_quality(
    model: nn.Module,
    tokenizer: object,
    prompt: str,
    num_tokens: int = 50,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Basic text quality check.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        num_tokens: Number of tokens to generate
        device: Device

    Returns:
        Dict with quality metrics
    """
    print(f"\n{'=' * 60}")
    print("Measuring Text Quality...")
    print(f"{'=' * 60}")

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Quality metrics
    num_words = len(generated_text.split())
    unique_words = len(set(generated_text.lower().split()))
    repetition_ratio = 1 - (unique_words / max(num_words, 1))

    print(f"Generated text: {generated_text[:100]}...")
    print(f"Total words: {num_words}")
    print(f"Unique words: {unique_words}")
    print(f"Repetition ratio: {repetition_ratio:.2f}")

    return {
        "generated_text": generated_text,
        "num_words": num_words,
        "unique_words": unique_words,
        "repetition_ratio": repetition_ratio,
    }


def evaluate_bubble(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    target_layers: List[int] = [3, 7, 11],
    thresholds_path: str = "calibración/layer_thresholds_progressive.json",
    output_path: str = "results/evaluation_report.json",
    device: str = "cuda",
    prompt: str = "El inventor del Bubble Transformer descubrió que la atención dispersa es análoga a",
    validation_text: str = None,
):
    """
    Run full evaluation suite.

    Args:
        model_name: Model name or path
        target_layers: Target layer indices
        thresholds_path: Path to calibration thresholds
        output_path: Path to save evaluation report
        device: Device
        prompt: Prompt for generation
        validation_text: Text for perplexity measurement
    """
    print("\n" + "=" * 60)
    print("BUBBLE TRANSFORMER V3 EVALUATION")
    print("=" * 60)

    # Check CUDA
    if device == "cuda" and not torch.cuda.is_available():
        print("\n[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    # Load thresholds
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
        print(f"[Info] Loaded calibration thresholds from {thresholds_path}")
    else:
        thresholds = {}
        print(f"[Warning] Thresholds file not found: {thresholds_path}")

    # Load model
    model, tokenizer = load_qwen_model(model_name, device)

    # Inject Bubble Attention
    model = inject_bubble_attention(
        model=model,
        target_layers=target_layers,
        thresholds=thresholds,
        default_C=32,
    )

    # Default validation text
    if validation_text is None:
        validation_text = """
        El Bubble Transformer es una arquitectura de atención que reemplaza el mecanismo
        tradicional de softmax por transporte óptimo entrópico. Esto permite una atención
        más dispersa y eficiente, reduciendo la complejidad computacional de O(N²) a O(N log C).
        La clave está en el coeficiente de viscosidad ε que controla el balance entre
        concentración y expressividad.
        """

    # Run evaluations
    results = {
        "model": model_name,
        "target_layers": target_layers,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1. Perplexity
    results["perplexity"] = measure_perplexity(
        model, tokenizer, validation_text, device
    )

    # 2. Latency
    results["latency"] = measure_latency(
        model, tokenizer, prompt, num_tokens=50, device=device
    )

    # 3. Memory
    results["memory"] = measure_memory_usage(
        model, tokenizer, prompt, num_tokens=50, device=device
    )

    # 4. Text Quality
    results["text_quality"] = measure_text_quality(
        model, tokenizer, prompt, num_tokens=50, device=device
    )

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_path}")

    # Print summary
    print(f"\nSUMMARY:")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Latency: {results['latency']['ms_per_token']:.2f} ms/token")
    print(f"  Peak Memory: {results['memory']['peak_memory_mb']:.2f} MB")
    print(f"  Repetition Ratio: {results['text_quality']['repetition_ratio']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Bubble Transformer V3 Performance"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model name or path",
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=[3, 7, 11],
        help="Target layer indices for injection",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="calibración/layer_thresholds_progressive.json",
        help="Path to calibration thresholds JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_report.json",
        help="Path to save evaluation report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="El inventor del Bubble Transformer descubrió que la atención dispersa es análoga a",
        help="Prompt for generation",
    )

    args = parser.parse_args()

    evaluate_bubble(
        model_name=args.model,
        target_layers=args.target_layers,
        thresholds_path=args.thresholds,
        output_path=args.output,
        device=args.device,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
