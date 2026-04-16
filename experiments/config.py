"""
Centralized Configuration — Bubble Transformer Experiment
==========================================================
Single source of truth for all experiment parameters.
Import this module instead of hardcoding values across files.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
class Qwen3_06B_Config:
    """Qwen3-0.6B-Base model configuration (28 layers, d_model=1024).

    Target layers: [0, 4, 8, 12, 16, 20, 24] (7 distributed layers)
    All 28 layers are full-attention (no DeltaNet in this model size).
    """

    model_name: str = "Qwen/Qwen3-0.6B-Base"
    num_layers: int = 28
    d_model: int = 1024
    num_attention_heads: int = 16
    num_kv_heads: int = 8  # Grouped Query Attention
    head_dim: int = 64  # d_model // num_attention_heads
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    max_length: int = 512
    batch_size: int = 4
    # 7 distributed layers across 28 total
    target_layers: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24])


@dataclass
class AttentionConfig:
    """PlateauAttention / Sinkhorn configuration."""

    # Default to Qwen3-0.6B dimensions (1024)
    # For Qwen 3.6, override with d_model=2048, num_heads=16, head_dim=128
    d_model: int = 1024
    num_heads: int = 16
    head_dim: int = 64  # d_model // num_heads
    tau_iters: int = 5  # Sinkhorn iterations
    dropout: float = 0.0


@dataclass
class EpsilonSweepConfig:
    """ε sweep configuration."""

    # Logarithmic sweep: from very concentrated (ε=0.001) to nearly uniform (ε=1.0)
    epsilon_values: List[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    # Qwen3-0.6B: 7 distributed layers across 28 total layers
    # [0, 4, 8, 12, 16, 20, 24] covers early, middle, and late layers
    target_layers: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24])
    # Legacy target for Qwen 3.6 (24-layer model)
    target_layers_legacy: List[int] = field(
        default_factory=lambda: [3, 7, 11, 15, 19, 23]
    )


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


VALID_COST_TYPES = ["l2_sq", "cosine", "dot_product", "mahalanobis", "mesh_learnable"]


@dataclass
class CostFunctionConfig:
    """Cost function configuration for PlateauAttention."""

    cost_type: str = "l2_sq"  # Default: L2 squared distance
    normalize: bool = True  # Normalize cost matrices for epsilon compatibility
    # Per-cost epsilon ranges (different cost functions have different scales)
    cost_epsilon_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "l2_sq": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            "cosine": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            "dot_product": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            "mahalanobis": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            "mesh_learnable": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        }
    )


@dataclass
class TensionConfig:
    """Dual-Head Tension configuration."""

    enabled: bool = False
    epsilon_low: float = 0.001  # Concentration head
    epsilon_high: float = 0.1  # Expressivity head
    alpha: float = 0.5  # Tension coefficient (0=pure-high, 1=pure-low)
    alpha_values: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )


@dataclass
class GOATConfig:
    """GOAT (Gated Optimal Attention Transport) configuration."""

    enabled: bool = False
    learn_gates: bool = True  # Enable gate learning
    gate_lr: float = 0.01  # Learning rate for gates
    gate_init: float = 1.0  # Initial gate value
    tied: bool = True  # Share gates across key groups


@dataclass
class LayerSelectionConfig:
    """Layer selection configuration."""

    enable_dual_head: bool = True  # Include dual-head comparison in layer selection
    alpha_values: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    comparison_metrics: List[str] = field(
        default_factory=lambda: [
            "effective_rank",
            "concentration_ratio",
            "intrinsic_dim_mle",
            "anisotropy_index",
        ]
    )


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
    cost_function: CostFunctionConfig = field(default_factory=CostFunctionConfig)
    tension: TensionConfig = field(default_factory=TensionConfig)
    goat: GOATConfig = field(default_factory=GOATConfig)
    layer_selection: LayerSelectionConfig = field(default_factory=LayerSelectionConfig)
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
    print(
        f"\nCost function: type={config.cost_function.cost_type}, normalize={config.cost_function.normalize}"
    )
    print(f"Cost epsilon ranges:")
    for ct, eps_range in config.cost_function.cost_epsilon_ranges.items():
        print(f"  {ct}: {eps_range}")
    print(f"\nEpsilon sweep: {config.epsilon_sweep.epsilon_values}")
    print(f"Target layers: {config.epsilon_sweep.target_layers}")
    print(f"\nTension: enabled={config.tension.enabled}")
    print(
        f"  epsilon_low={config.tension.epsilon_low}, epsilon_high={config.tension.epsilon_high}"
    )
    print(f"  alpha={config.tension.alpha}, alpha_values={config.tension.alpha_values}")
    print(
        f"\nLayer Selection: enable_dual_head={config.layer_selection.enable_dual_head}"
    )
    print(f"  alpha_values={config.layer_selection.alpha_values}")
    print(f"  comparison_metrics={config.layer_selection.comparison_metrics}")
    print(f"\nMetrics thresholds:")
    print(f"  Min effective rank: {config.metrics.min_effective_rank}")
    print(f"  Max anisotropy: {config.metrics.max_anisotropy}")
    print(f"  Min intrinsic dim: {config.metrics.min_intrinsic_dim}")
    print(f"\nPaths:")
    print(f"  Embeddings: {config.paths.embeddings_dir}")
    print(f"  Results: {config.paths.results_dir}")
    print(f"  Plots: {config.paths.plots_dir}")
    print(f"  Corpus: {config.paths.corpus_path}")
