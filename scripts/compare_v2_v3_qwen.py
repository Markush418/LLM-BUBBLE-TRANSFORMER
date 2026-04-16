"""
Compare V2 (Sinkhorn) vs V3 (SDOT) on Qwen Model
================================================

Comprehensive benchmark comparing:
- V2: PlateauAttentionMechanism (Sinkhorn-Knopp iterative)
- V3: SDOTAttention (Semi-Discrete Optimal Transport)

Measures:
1. Throughput (tokens/second)
2. Memory (peak VRAM in MB)
3. Effective Rank (representation quality)
4. Sparsity (attention concentration)
5. Output Similarity (cosine similarity between V2 and V3 outputs)
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.sdot_attention import SDOTAttention
from experiments.plateau_attention import PlateauAttentionMechanism


# =============================================================================
# Qwen Configuration
# =============================================================================

QWEN_CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "d_model": 896,
    "num_heads": 14,
    "head_dim": 64,
    "num_layers": 24,
}


# =============================================================================
# V2 Attention Wrapper (PyTorch <-> NumPy bridge)
# =============================================================================


class V2AttentionWrapper(nn.Module):
    """
    Wrapper for PlateauAttentionMechanism (V2) to work with PyTorch tensors.

    Handles conversion between PyTorch tensors and NumPy arrays.
    Uses PlateauAttentionMechanism internally (NumPy-based Sinkhorn).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        epsilon: float = 0.01,
        tau_iters: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.epsilon = epsilon

        # Create NumPy-based V2 attention
        self.plateau_attn = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon,
            tau_iters=tau_iters,
            seed=seed,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with PyTorch tensor.

        Args:
            x: [B, N, d_model] PyTorch tensor
            return_attention: If True, return attention matrix

        Returns:
            output: [B, N, d_model] PyTorch tensor
            attention: [B, heads, N, N] (if return_attention=True)
        """
        # Convert to NumPy
        x_np = x.detach().cpu().numpy().astype(np.float32)

        # Run V2 (NumPy)
        if return_attention:
            output_np, attention_np = self.plateau_attn.forward(
                x_np, return_attention=True
            )
        else:
            output_np = self.plateau_attn.forward(x_np)
            attention_np = None

        # Convert back to PyTorch
        output = torch.from_numpy(output_np).to(x.device)

        if return_attention:
            attention = torch.from_numpy(attention_np).to(x.device)
            return output, attention

        return output, None


# =============================================================================
# Metrics Functions
# =============================================================================


def compute_effective_rank(attention: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Compute effective rank of attention matrix.

    Effective rank = number of singular values needed to explain
    (1 - threshold) of total variance.

    Args:
        attention: [B, heads, N, N] or [B, N, N] attention matrix
        threshold: Variance threshold (default: 0.01 = 99% explained)

    Returns:
        effective_rank: Average effective rank across batch and heads
    """
    if attention.dim() == 4:
        # [B, heads, N, N] -> flatten batch and heads
        B, H, N, _ = attention.shape
        attention = attention.reshape(B * H, N, N)

    effective_ranks = []

    for i in range(attention.shape[0]):
        # SVD
        try:
            U, S, V = torch.linalg.svd(attention[i], full_matrices=False)
        except RuntimeError:
            # Fallback to CPU if SVD fails on CUDA
            attn_cpu = attention[i].cpu()
            U, S, V = torch.linalg.svd(attn_cpu, full_matrices=False)

        # Normalize singular values
        S_normalized = S / S.sum()

        # Cumulative variance
        cumvar = torch.cumsum(S_normalized, dim=0)

        # Find index where cumvar >= 1 - threshold
        idx = (cumvar >= (1.0 - threshold)).nonzero()
        if len(idx) > 0:
            effective_rank = idx[0].item() + 1
        else:
            effective_rank = len(S)

        effective_ranks.append(effective_rank)

    return float(np.mean(effective_ranks))


def compute_sparsity(attention: torch.Tensor, threshold: float = 1e-5) -> float:
    """
    Compute sparsity of attention matrix.

    Sparsity = fraction of attention weights < threshold.

    Args:
        attention: [B, heads, N, N] or [B, N, N] attention matrix
        threshold: Sparsity threshold (default: 1e-5)

    Returns:
        sparsity: Average sparsity across batch and heads (0.0 to 1.0)
    """
    if attention.dim() == 4:
        # [B, heads, N, N]
        B, H, N, _ = attention.shape
        total_elements = B * H * N * N
    else:
        # [B, N, N]
        B, N, _ = attention.shape
        total_elements = B * N * N

    sparse_elements = (attention < threshold).sum().item()

    return float(sparse_elements / total_elements)


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two tensors.

    Args:
        a: [B, N, d_model] tensor
        b: [B, N, d_model] tensor

    Returns:
        similarity: Cosine similarity (0.0 to 1.0)
    """
    # Flatten
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    # Cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1
    )

    return float(similarity.item())


# =============================================================================
# Qwen Comparator
# =============================================================================


class QwenComparator:
    """
    Compares V2 (Sinkhorn) and V3 (SDOT) on Qwen model embeddings.

    Workflow:
    1. Load Qwen model (or use synthetic embeddings)
    2. Generate embeddings from text prompts
    3. Run V2 and V3 attention
    4. Measure throughput, memory, effective rank, sparsity, similarity
    5. Generate comparison report
    """

    def __init__(
        self,
        d_model: int = 896,
        num_heads: int = 14,
        epsilon: float = 0.01,
        num_centroids: int = 32,
        device: str = "cuda",
        use_real_model: bool = False,
        model_name: str = "Qwen/Qwen2.5-0.5B",
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.epsilon = epsilon
        self.num_centroids = num_centroids
        self.device = device
        self.use_real_model = use_real_model
        self.model_name = model_name

        # Create V2 (Sinkhorn)
        self.v2_attention = V2AttentionWrapper(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon,
            tau_iters=5,
        )

        # Create V3 (SDOT)
        self.v3_attention = SDOTAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_centroids=num_centroids,
            use_baroreceptor=False,
        )

        if device == "cuda" and torch.cuda.is_available():
            self.v3_attention = self.v3_attention.to(device)

        # Load Qwen model if requested
        self.tokenizer = None
        self.model = None

        if use_real_model:
            self._load_qwen_model()

    def _load_qwen_model(self):
        """Load Qwen model from HuggingFace."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"[QwenComparator] Loading {self.model_name}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.device == "cpu" and self.model is not None:
                self.model = self.model.to(self.device)

            self.model.eval()

            print(f"[QwenComparator] Model loaded successfully")

        except Exception as e:
            print(f"[QwenComparator] Failed to load model: {e}")
            print("[QwenComparator] Falling back to synthetic embeddings")
            self.use_real_model = False
            self.tokenizer = None
            self.model = None

    def generate_embeddings(
        self,
        prompts: List[str],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Generate embeddings from text prompts.

        Args:
            prompts: List of text prompts
            batch_size: Batch size for processing

        Returns:
            embeddings: [len(prompts), seq_len, d_model]
        """
        if self.use_real_model and self.model is not None:
            return self._generate_real_embeddings(prompts, batch_size)
        else:
            return self._generate_synthetic_embeddings(prompts)

    def _generate_real_embeddings(
        self,
        prompts: List[str],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generate embeddings from real Qwen model."""
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )

                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get hidden states
                outputs = self.model(**inputs, output_hidden_states=True)

                # Use last hidden state
                hidden_states = outputs.hidden_states[-1]  # [B, N, d_model]

                all_embeddings.append(hidden_states.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _generate_synthetic_embeddings(
        self,
        prompts: List[str],
    ) -> torch.Tensor:
        """Generate synthetic embeddings (for testing without model)."""
        # Use fixed seq_len for consistency
        seq_len = 64
        batch_size = len(prompts)

        # Generate random embeddings with some structure
        torch.manual_seed(42)
        embeddings = torch.randn(batch_size, seq_len, self.d_model)

        # Add some structure based on prompt length
        for i, prompt in enumerate(prompts):
            scale = len(prompt) / 100.0
            embeddings[i] *= 1.0 + scale

        return embeddings

    def benchmark_single(
        self,
        x: torch.Tensor,
        num_runs: int = 5,
        warmup_runs: int = 2,
    ) -> Dict:
        """
        Benchmark V2 and V3 on a single input.

        Args:
            x: [B, N, d_model] input tensor
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Dict with benchmark results
        """
        B, N, D = x.shape

        # Move to device
        x_device = x.to(self.device) if self.device == "cuda" else x

        # Warmup
        for _ in range(warmup_runs):
            _ = self.v2_attention.forward(x_device)
            _ = self.v3_attention(x_device)

            if self.device == "cuda":
                torch.cuda.synchronize()

        # Benchmark V2
        times_v2 = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output_v2, attention_v2 = self.v2_attention.forward(
                x_device, return_attention=True
            )
            end = time.perf_counter()
            times_v2.append((end - start) * 1000)  # ms

        # Benchmark V3
        times_v3 = []
        for _ in range(num_runs):
            if self.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            output_v3, attention_v3 = self.v3_attention(
                x_device, return_assignments=True
            )

            if self.device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times_v3.append((end - start) * 1000)  # ms

        # Compute metrics
        avg_v2_time = np.mean(times_v2)
        avg_v3_time = np.mean(times_v3)

        # Memory (V2 is NumPy, estimate; V3 is PyTorch, measure)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = self.v3_attention(x_device)
            torch.cuda.synchronize()
            v3_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            # Estimate from attention matrix size
            v3_memory_mb = (B * self.num_heads * N * N * 4) / (1024 * 1024)

        # V2 memory (NumPy, estimate)
        v2_memory_mb = (B * self.num_heads * N * N * 4) / (1024 * 1024)

        # Effective rank (V2)
        eff_rank_v2 = compute_effective_rank(attention_v2)

        # Sparsity (V2)
        sparsity_v2 = compute_sparsity(attention_v2)

        # Output similarity
        output_v2_cpu = (
            output_v2.cpu() if output_v2.device.type == "cuda" else output_v2
        )
        output_v3_cpu = (
            output_v3.cpu() if output_v3.device.type == "cuda" else output_v3
        )
        similarity = compute_cosine_similarity(output_v2_cpu, output_v3_cpu)

        # Hard support (V3)
        if attention_v3 is not None:
            from models.v3_core import compute_hard_support

            hard_support = compute_hard_support(attention_v3).mean().item()
        else:
            hard_support = 0.0

        return {
            "batch_size": B,
            "seq_length": N,
            "v2": {
                "time_ms": float(avg_v2_time),
                "time_std_ms": float(np.std(times_v2)),
                "memory_mb": float(v2_memory_mb),
                "effective_rank": float(eff_rank_v2),
                "sparsity": float(sparsity_v2),
            },
            "v3": {
                "time_ms": float(avg_v3_time),
                "time_std_ms": float(np.std(times_v3)),
                "memory_mb": float(v3_memory_mb),
                "hard_support": float(hard_support),
                "num_centroids": self.num_centroids,
            },
            "comparison": {
                "speedup": float(avg_v2_time / avg_v3_time) if avg_v3_time > 0 else 0.0,
                "memory_reduction": float(v2_memory_mb / v3_memory_mb)
                if v3_memory_mb > 0
                else 0.0,
                "output_similarity": float(similarity),
            },
        }

    def run_comparison(
        self,
        prompts: List[str],
        num_runs: int = 5,
        warmup_runs: int = 2,
    ) -> Dict:
        """
        Run full comparison across all prompts.

        Args:
            prompts: List of text prompts
            num_runs: Number of benchmark runs per prompt
            warmup_runs: Number of warmup runs

        Returns:
            Dict with full comparison results
        """
        print("\n" + "=" * 70)
        print("QWEN V2 vs V3 COMPARISON")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"d_model: {self.d_model}, num_heads: {self.num_heads}")
        print(f"V2 epsilon: {self.epsilon}")
        print(f"V3 num_centroids: {self.num_centroids}")
        print("=" * 70)

        # Generate embeddings
        print("\n[Step 1] Generating embeddings...")
        embeddings = self.generate_embeddings(prompts)
        print(f"Generated embeddings: {embeddings.shape}")

        # Run benchmarks
        print("\n[Step 2] Running benchmarks...")
        all_results = []

        for i, prompt in enumerate(prompts):
            print(f'\n[Prompt {i + 1}/{len(prompts)}] "{prompt[:50]}..."')

            # Get single embedding
            x = embeddings[i : i + 1]  # [1, N, d_model]

            # Benchmark
            result = self.benchmark_single(x, num_runs, warmup_runs)
            result["prompt"] = prompt
            all_results.append(result)

            # Print intermediate results
            print(
                f"  V2 time: {result['v2']['time_ms']:.2f} ± {result['v2']['time_std_ms']:.2f} ms"
            )
            print(
                f"  V3 time: {result['v3']['time_ms']:.2f} ± {result['v3']['time_std_ms']:.2f} ms"
            )
            print(f"  Speedup: {result['comparison']['speedup']:.2f}x")
            print(f"  Similarity: {result['comparison']['output_similarity']:.4f}")

        # Aggregate results
        avg_speedup = np.mean([r["comparison"]["speedup"] for r in all_results])
        avg_memory_reduction = np.mean(
            [r["comparison"]["memory_reduction"] for r in all_results]
        )
        avg_similarity = np.mean(
            [r["comparison"]["output_similarity"] for r in all_results]
        )
        avg_v2_time = np.mean([r["v2"]["time_ms"] for r in all_results])
        avg_v3_time = np.mean([r["v3"]["time_ms"] for r in all_results])
        avg_eff_rank = np.mean([r["v2"]["effective_rank"] for r in all_results])
        avg_sparsity = np.mean([r["v2"]["sparsity"] for r in all_results])

        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        print("\nV2 (Sinkhorn):")
        print(f"  Time: {avg_v2_time:.2f} ms")
        print(f"  Memory: {all_results[0]['v2']['memory_mb']:.2f} MB")
        print(f"  Eff Rank: {avg_eff_rank:.1f}")
        print(f"  Sparsity: {avg_sparsity:.2%}")

        print("\nV3 (SDOT):")
        print(f"  Time: {avg_v3_time:.2f} ms")
        print(f"  Memory: {all_results[0]['v3']['memory_mb']:.2f} MB")
        print(f"  Hard Support: {all_results[0]['v3']['hard_support']:.0f}")
        print(f"  Centroids: {self.num_centroids}")

        print("\nComparison:")
        print(f"  Speedup: {avg_speedup:.2f}x")
        print(f"  Memory Reduction: {avg_memory_reduction:.2f}x")
        print(f"  Output Similarity: {avg_similarity:.4f}")

        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)

        if avg_speedup > 1.0:
            print(f"[PASS] V3 (SDOT) is {avg_speedup:.2f}x FASTER than V2 (Sinkhorn)")
        else:
            print(
                f"[WARN] V3 (SDOT) is {1 / avg_speedup:.2f}x SLOWER than V2 (Sinkhorn)"
            )

        if avg_memory_reduction > 1.0:
            print(f"[PASS] V3 uses {avg_memory_reduction:.2f}x LESS MEMORY than V2")
        else:
            print(f"[WARN] V3 uses {1 / avg_memory_reduction:.2f}x MORE MEMORY than V2")

        if avg_similarity > 0.5:
            print(
                f"[PASS] Outputs are {avg_similarity * 100:.1f}% similar (high fidelity)"
            )
        else:
            print(
                f"[WARN] Outputs are {avg_similarity * 100:.1f}% similar (low fidelity)"
            )

        print("=" * 70)

        if avg_speedup > 1.0:
            print(f"[PASS] V3 (SDOT) is {avg_speedup:.2f}x FASTER than V2 (Sinkhorn)")
        else:
            print(
                f"[WARN] V3 (SDOT) is {1 / avg_speedup:.2f}x SLOWER than V2 (Sinkhorn)"
            )

        if avg_memory_reduction > 1.0:
            print(f"[PASS] V3 uses {avg_memory_reduction:.2f}x LESS MEMORY than V2")
        else:
            print(f"[WARN] V3 uses {1 / avg_memory_reduction:.2f}x MORE MEMORY than V2")

        if avg_similarity > 0.5:
            print(
                f"[PASS] Outputs are {avg_similarity * 100:.1f}% similar (high fidelity)"
            )
        else:
            print(
                f"[WARN] Outputs are {avg_similarity * 100:.1f}% similar (low fidelity)"
            )

        print("=" * 70)

        # Return full results
        return {
            "config": {
                "model_name": self.model_name,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "epsilon": self.epsilon,
                "num_centroids": self.num_centroids,
                "device": self.device,
                "use_real_model": self.use_real_model,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            },
            "prompts": prompts,
            "results": all_results,
            "summary": {
                "avg_speedup": float(avg_speedup),
                "avg_memory_reduction": float(avg_memory_reduction),
                "avg_similarity": float(avg_similarity),
                "avg_v2_time_ms": float(avg_v2_time),
                "avg_v3_time_ms": float(avg_v3_time),
                "avg_effective_rank": float(avg_eff_rank),
                "avg_sparsity": float(avg_sparsity),
            },
        }


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare V2 (Sinkhorn) vs V3 (SDOT) on Qwen model"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model name (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=896,
        help="Model dimension (default: 896)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=14,
        help="Number of attention heads (default: 14)",
    )

    # V2 parameters
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="V2 epsilon parameter (default: 0.01)",
    )

    # V3 parameters
    parser.add_argument(
        "--num-centroids",
        type=int,
        default=32,
        help="V3 number of centroids (default: 32)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="results/v2_v3_comparison.json",
        help="Output JSON path (default: results/v2_v3_comparison.json)",
    )

    # Mode
    parser.add_argument(
        "--use-real-model",
        action="store_true",
        help="Use real Qwen model (requires GPU and transformers)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic embeddings (for testing without model)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        args.device = "cpu"

    # Determine mode
    use_real_model = args.use_real_model and not args.synthetic

    # Test prompts
    prompts = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
    ]

    # Create comparator
    comparator = QwenComparator(
        d_model=args.d_model,
        num_heads=args.num_heads,
        epsilon=args.epsilon,
        num_centroids=args.num_centroids,
        device=args.device,
        use_real_model=use_real_model,
        model_name=args.model,
    )

    # Run comparison
    results = comparator.run_comparison(
        prompts=prompts,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Results saved to {output_path}]")


if __name__ == "__main__":
    main()
