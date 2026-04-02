# Spec: Plan A+B Combined — Embedding Geometry Map + ε Sweet Spot

**Date**: 2026-04-01  
**Project**: LLM-BUBBLE / Bubble Transformer Research  
**Author**: Sisyphus (automate.dev)  
**Status**: Design Complete

---

## 1. Objective

Execute **Plan A** (embedding geometry mapping under Sinkhorn vs Softmax) and **Plan B** (optimal ε sweep for concentration) as a single unified experimental pipeline.

### Success Criteria

1. **Plan A**: Produce per-layer comparison of embedding distributions (Softmax baseline vs Plateau/Sinkhorn) across 6+ layers of Qwen 3.6
2. **Plan B**: Identify the ε value (or narrow range) that maximizes embedding concentration without representational collapse
3. **Deliverable**: Heatmap of ε × layer showing effective rank, concentration ratio, and intrinsic dimensionality

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Qwen 3.6 (Frozen)                    │
│  Input: batch of texts → extract embeddings per layer │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
   ┌────▼────┐             ┌─────▼─────┐
   │Softmax  │             │Plateau    │
   │(baseline│             │Attention  │
   │  Qwen)  │             │(Sinkhorn) │
   └────┬────┘             └─────┬─────┘
        │                         │
        │              ┌──────────┼──────────┐
        │              │          │          │
        │           ε=0.01    ε=0.05    ε=0.1 ... ε=1.0
        │              │          │          │
        └──────┬───────┴──────────┴──────────┘
               │
    ┌──────────▼──────────┐
    │  Metrics per layer   │
    │  • Effective Rank    │
    │  • Intrinsic Dim     │
    │  • Anisotropy Index  │
    │  • Pairwise Dist     │
    │  • Concentration     │
    └──────────┬───────────┘
               │
    ┌──────────▼──────────┐
    │  Visualization       │
    │  • t-SNE/UMAP        │
    │  • Pareto Curves     │
    │  • Heatmaps ε×layer  │
    └─────────────────────┘
```

---

## 3. Components

### 3.1 Embedding Extractor (`extract_embeddings.py`)

**Responsibility**: Load Qwen 3.6 (frozen), run forward passes, extract hidden states at every layer.

**Inputs**:
- Model: `Qwen/Qwen3.6-Plus` or any available variant from HuggingFace
- Text corpus: 50-100 diverse prompts (code, math, prose, dialogue, reasoning)
- Batch size: 4-8 (VRAM dependent)

**Outputs**:
- `embeddings/softmax/layer_{N}.pt` — baseline Qwen embeddings per layer
- `embeddings/raw_input.pt` — pre-attention embeddings (same for all variants)

**Key design**:
```python
class QwenEmbeddingExtractor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        # Register forward hooks on each transformer layer
        self.layer_outputs = {}
        for i, layer in enumerate(self.model.model.layers):
            layer.register_forward_hook(
                lambda mod, inp, out, idx=i: self._capture(idx, out)
            )
    
    def extract(self, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        with torch.no_grad():
            self.model(input_ids)
        return self.layer_outputs  # {layer_idx: hidden_states [B, N, D]}
```

**Constraints**:
- Model stays frozen (no gradients)
- bfloat16 for memory efficiency
- Hooks capture hidden states BEFORE the residual addition (to isolate attention output)

---

### 3.2 PlateauAttentionMechanism (`plateau_attention.py`)

**Responsibility**: Drop-in replacement for Qwen's softmax attention using Sinkhorn-Knopp.

**Implementation** (extends the existing code in `texto.txt`):

```python
class PlateauAttentionMechanism(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        epsilon: float = 0.1,
        tau_iters: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.epsilon = epsilon
        self.tau_iters = tau_iters
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Geometric Cost Matrix: C_ij = ||Q_i - K_j||^2
        C = torch.cdist(Q, K, p=2).pow(2)  # [B, heads, N, N]
        
        # Log-domain Sinkhorn
        log_S = -C / self.epsilon
        
        if mask is not None:
            log_S = log_S.masked_fill(mask == 0, float('-inf'))
        
        u = torch.zeros(B, self.num_heads, N, device=x.device, dtype=x.dtype)
        v = torch.zeros(B, self.num_heads, N, device=x.device, dtype=x.dtype)
        
        for _ in range(self.tau_iters):
            u = -torch.logsumexp(log_S + v.unsqueeze(-2), dim=-1)
            v = -torch.logsumexp(log_S + u.unsqueeze(-1), dim=-2)
        
        A_minimal = torch.exp(log_S + u.unsqueeze(-1) + v.unsqueeze(-2))
        
        output = torch.matmul(A_minimal, V)
        output = output.transpose(1, 2).reshape(B, N, D)
        return output
```

**Key differences from baseline**:
- Replaces `softmax(QK^T / sqrt(d))` with Sinkhorn-normalized cost matrix
- Multi-head support added
- Log-domain for numerical stability

---

### 3.3 ε Sweep Controller (`epsilon_sweep.py`)

**Responsibility**: Orchestrate the ε sweep across all target layers.

**ε values** (logarithmic sweep):
```python
EPSILON_VALUES = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
```

**Target layers** (Qwen 3.6 full-attention layers, every 4th):
```python
TARGET_LAYERS = [3, 7, 11, 15, 19, 23]  # 6 layers across the depth
```

**Execution flow**:
```
for ε in EPSILON_VALUES:
    for layer_idx in TARGET_LAYERS:
        1. Load raw embeddings at layer_idx
        2. Apply PlateauAttention with ε
        3. Compute all metrics
        4. Store results in results/epsilon_{ε}_layer_{layer_idx}.json
```

---

### 3.4 Metrics Engine (`metrics.py`)

**Responsibility**: Compute all concentration and geometry metrics.

#### Metric 1: Effective Rank (SVD-based)
```python
def effective_rank(embeddings: torch.Tensor) -> float:
    """
    Computes the effective rank of the embedding matrix.
    Effective rank = exp(H(p)) where p = σ_i / sum(σ_j)
    σ_i are singular values from SVD.
    """
    _, S, _ = torch.svd(embeddings.reshape(-1, embeddings.shape[-1]).float())
    p = S / S.sum()
    p = p[p > 1e-10]  # filter numerical zeros
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()
```

#### Metric 2: Intrinsic Dimensionality (MLE estimator)
```python
def intrinsic_dimension_mle(embeddings: torch.Tensor, k: int = 10) -> float:
    """
    Maximum Likelihood Estimator for intrinsic dimensionality.
    Based on Levina & Bickel (2004).
    """
    # Compute k-NN distances
    dists = torch.cdist(embeddings, embeddings)
    dists = torch.sort(dists, dim=1).values[:, 1:k+1]  # exclude self
    ratios = torch.log(dists[:, -1:] / dists[:, :-1] + 1e-10)
    d_hat = 1.0 / ratios.mean()
    return d_hat.item()
```

#### Metric 3: Anisotropy Index
```python
def anisotropy_index(embeddings: torch.Tensor) -> float:
    """
    Ratio of max eigenvalue to sum of eigenvalues.
    1.0 = fully anisotropic (one direction dominates)
    1/d = perfectly isotropic
    """
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = torch.matmul(centered.T, centered) / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(cov.float())
    return (eigenvalues[-1] / eigenvalues.sum()).item()
```

#### Metric 4: Pairwise Distance Distribution
```python
def pairwise_distance_stats(embeddings: torch.Tensor) -> dict:
    """
    Compute mean, std, skewness of pairwise distances.
    High concentration → low mean, low std.
    """
    dists = torch.pdist(embeddings.reshape(-1, embeddings.shape[-1]).float())
    return {
        "mean": dists.mean().item(),
        "std": dists.std().item(),
        "min": dists.min().item(),
        "max": dists.max().item(),
        "median": dists.median().item(),
    }
```

#### Metric 5: Concentration Ratio
```python
def concentration_ratio(attention_matrix: torch.Tensor) -> float:
    """
    Fraction of non-negligible entries in the attention matrix.
    Lower = more concentrated (sparser attention).
    """
    threshold = 1.0 / attention_matrix.shape[-1]  # uniform baseline
    active = (attention_matrix > threshold).float().sum()
    total = attention_matrix.numel()
    return (active / total).item()
```

---

### 3.5 Visualization Engine (`visualize.py`)

**Responsibility**: Generate all plots and heatmaps.

**Outputs**:
1. **t-SNE/UMAP plots**: Per-layer, per-ε embedding scatter (2D projection)
2. **Effective Rank curves**: Layer index vs effective rank, one line per ε
3. **Concentration heatmaps**: ε (x-axis) × layer (y-axis), color = concentration ratio
4. **Pareto frontier**: Concentration vs effective rank scatter (find the sweet spot)
5. **Anisotropy vs ε**: Line plot showing collapse threshold

---

## 4. Data Flow

```
1. Load Qwen 3.6 → freeze → register hooks
2. Run forward pass on corpus → save softmax baseline embeddings
3. For each ε in [0.001, ..., 1.0]:
   a. For each layer in [3, 7, 11, 15, 19, 23]:
      i. Apply PlateauAttention with ε to raw embeddings
      ii. Compute 5 metrics
      iii. Save results
4. Aggregate all results → generate visualizations
5. Identify ε sweet spot → output recommendation
```

---

## 5. File Structure

```
LLM-BUBBLE/
├── experiments/
│   ├── extract_embeddings.py      # Component 3.1
│   ├── plateau_attention.py       # Component 3.2
│   ├── epsilon_sweep.py           # Component 3.3
│   ├── metrics.py                 # Component 3.4
│   ├── visualize.py               # Component 3.5
│   └── run_experiment.py          # Main orchestrator
├── embeddings/
│   ├── softmax/                   # Baseline Qwen embeddings
│   └── plateau/                   # Plateau attention embeddings
├── results/
│   ├── metrics_summary.csv        # All metrics in tabular form
│   ├── epsilon_sweep.json         # Full sweep results
│   └── sweet_spot_analysis.md     # ε recommendation
├── plots/
│   ├── tsne_layer_*.png
│   ├── effective_rank_curves.png
│   ├── concentration_heatmap.png
│   ├── pareto_frontier.png
│   └── anisotropy_vs_epsilon.png
└── docs/superpowers/specs/
    └── 2026-04-01-bubble-embedding-geometry-design.md  (this file)
```

---

## 6. Error Handling

| Failure Mode | Recovery |
|---|---|
| OOM during forward pass | Reduce batch size, use `device_map="auto"` with offloading |
| Sinkhorn divergence (ε too small) | Clamp ε ≥ 0.001, log warning, skip that ε value |
| NaN in embeddings | Switch to float32 for that layer, log warning |
| Model download fails | Cache locally, use offline mode |

---

## 7. Dependencies

```
torch>=2.1.0
transformers>=4.40.0
accelerate>=0.27.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0  # for t-SNE
umap-learn>=0.5.0    # for UMAP
tqdm>=4.65.0
```

---

## 8. Execution

```bash
# Single command to run everything
python experiments/run_experiment.py \
    --model "Qwen/Qwen3.6-Plus" \
    --device "cuda" \
    --batch-size 4 \
    --corpus "data/test_corpus.jsonl" \
    --output-dir "results/"

# Or step by step:
python experiments/extract_embeddings.py --model "Qwen/Qwen3.6-Plus"
python experiments/epsilon_sweep.py --epsilon-range 0.001 1.0 --steps 9
python experiments/visualize.py --results-dir "results/"
```

---

## 9. Expected Timeline

| Step | Estimated Time |
|------|---------------|
| Setup + dependency install | 15 min |
| Embedding extraction (6 layers × 50 texts) | 30-60 min |
| ε sweep (9 values × 6 layers) | 45-90 min |
| Metrics computation | 10 min |
| Visualization generation | 5 min |
| **Total** | **~2-3 hours** |

---

## 10. Definition of Done

- [ ] All 5 metrics computed for every (ε, layer) pair
- [ ] Heatmap of ε × layer concentration generated
- [ ] t-SNE/UMAP visualizations for at least 3 representative layers
- [ ] Pareto frontier plot identifying the ε sweet spot
- [ ] Written recommendation: "Use ε = X for layers Y-Z"
- [ ] Results saved in `results/` directory with full reproducibility
