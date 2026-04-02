"""
Centralized Configuration — Bubble Transformer Experiment
==========================================================
Single source of truth for all experiment parameters.
Import this module instead of hardcoding values across files.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Qwen 3.6 model configuration."""

    model_name: str = "Qwen/Qwen3.6-Plus"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"  # bfloat16 for memory efficiency
    device: str = "cuda"
    max_length: int = 512
    batch_size: int = 4


@dataclass
class AttentionConfig:
    """PlateauAttention / Sinkhorn configuration."""

    d_model: int = 2048
    num_heads: int = 16
    head_dim: int = 128  # d_model // num_heads
    tau_iters: int = 5  # Sinkhorn iterations
    dropout: float = 0.0


@dataclass
class EpsilonSweepConfig:
    """ε sweep configuration."""

    # Logarithmic sweep: from very concentrated (ε=0.001) to nearly uniform (ε=1.0)
    epsilon_values: List[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    # Qwen 3.6 full-attention layers (every 4th layer)
    target_layers: List[int] = field(default_factory=lambda: [3, 7, 11, 15, 19, 23])


@dataclass
class MetricsConfig:
    """Metrics computation configuration."""

    k_neighbors: int = 10  # For intrinsic dimensionality MLE
    concentration_threshold_factor: float = 1.0  # Multiplier for 1/N baseline
    min_effective_rank: float = 50.0  # Minimum acceptable effective rank
    max_anisotropy: float = 0.5  # Maximum acceptable anisotropy index
    min_intrinsic_dim: float = 20.0  # Minimum acceptable intrinsic dimensionality


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    dpi: int = 200
    tsne_perplexity: int = 30
    tsne_max_samples: int = 500  # Max points for t-SNE
    color_palette: str = "viridis"
    figsize_standard: tuple = (12, 7)
    figsize_large: tuple = (16, 12)


@dataclass
class PathConfig:
    """Directory paths."""

    base_dir: str = "."
    experiments_dir: str = "experiments"
    data_dir: str = "data"
    embeddings_dir: str = "embeddings"
    softmax_dir: str = "embeddings/softmax"
    plateau_dir: str = "embeddings/plateau"
    results_dir: str = "results"
    plots_dir: str = "plots"
    corpus_path: str = "data/test_corpus.jsonl"
    sweep_results_path: str = "results/epsilon_sweep.json"
    sweet_spot_report_path: str = "results/sweet_spot_analysis.md"


@dataclass
class ExperimentConfig:
    """Master configuration for the full experiment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    epsilon_sweep: EpsilonSweepConfig = field(default_factory=EpsilonSweepConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Experiment flags
    skip_extraction: bool = False
    skip_visualization: bool = False
    device_override: Optional[str] = None  # If set, overrides model.device

    def get_device(self) -> str:
        """Get the effective device."""
        return self.device_override or self.model.device


# ─── Singleton instance ─────────────────────────────────────────────────────

_default_config = None


def get_config() -> ExperimentConfig:
    """Get the default experiment configuration."""
    global _default_config
    if _default_config is None:
        _default_config = ExperimentConfig()
    return _default_config


def reset_config():
    """Reset to default configuration."""
    global _default_config
    _default_config = None


if __name__ == "__main__":
    # Print current configuration
    config = get_config()
    print("=" * 60)
    print("  Bubble Transformer — Experiment Configuration")
    print("=" * 60)
    print(f"\nModel: {config.model.model_name}")
    print(f"Device: {config.get_device()}")
    print(f"Batch size: {config.model.batch_size}")
    print(f"Max length: {config.model.max_length}")
    print(
        f"\nAttention: d_model={config.attention.d_model}, heads={config.attention.num_heads}"
    )
    print(f"Tau iterations: {config.attention.tau_iters}")
    print(f"\nEpsilon sweep: {config.epsilon_sweep.epsilon_values}")
    print(f"Target layers: {config.epsilon_sweep.target_layers}")
    print(f"\nMetrics thresholds:")
    print(f"  Min effective rank: {config.metrics.min_effective_rank}")
    print(f"  Max anisotropy: {config.metrics.max_anisotropy}")
    print(f"  Min intrinsic dim: {config.metrics.min_intrinsic_dim}")
    print(f"\nPaths:")
    print(f"  Embeddings: {config.paths.embeddings_dir}")
    print(f"  Results: {config.paths.results_dir}")
    print(f"  Plots: {config.paths.plots_dir}")
    print(f"  Corpus: {config.paths.corpus_path}")
