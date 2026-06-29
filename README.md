# LLM-BUBBLE — Bubble Transformer Research

> **Hybrid Attention Architecture: DeltaNet + SIRI + Power Diagram**
> Plan A+B: Mapear distribución de embeddings bajo Sinkhorn vs Softmax y encontrar el coeficiente de viscosidad óptimo.

---

## 🎯 Objetivo

Determinar **qué embeddings** de Qwen 3.6 se concentran mejor en la "sección de gravedad" del Bubble Transformer, y **qué valor de ε** (viscosidad) maximiza esta concentración sin colapso representacional.

### Plan A — Embedding Geometry Map
Comparar cómo se distribuyen los embeddings de Qwen 3.6 bajo:
- **Softmax** (baseline, atención original de Qwen)
- **DeltaNet** (atención lineal con delta rule, default post-SDOT)
- **PlateauAttention/SIRI** (Sinkhorn doubly-stochastic preservando ε)

### Plan B — ε Sweet Spot
Encontrar el rango de ε que produce máxima concentración de embeddings sin colapso dimensional.

---

## 🏗️ Arquitectura Híbrida (post-SDOT, junio 2026)

[DEFINITION] La nueva arquitectura del Bubble Transformer combina:

1. **DeltaNet** (Yang et al. 2024, arxiv:2406.06484) — atención lineal O(N) con delta rule para asociative recall
2. **SIRI post-processing** (Sinkhorn-Knopp log-domain) — refinamiento doubly-stochastic opt-in
3. **Power Diagram ψ** — bias en `log_S = -C/ε + ψ` para Laguerre tessellation
4. **SIRI-Soft variants** (NEW, June 2026) — soft blend / chiller / sparse variants that preserve peakedness
5. **Hybrid interpolation** — `out = λ·out_delta + (1-λ)·out_siri`

```python
from hybrid_attention import HybridAttention

attn = HybridAttention(
    d_model=1024,            # Qwen3-0.6B dimensions
    num_heads=16,            # Qwen3 attention heads
    epsilon=0.1,             # SIRI bandwidth
    lam=0.5,                 # 0.5 = balanced hybrid
    siri_mode="soft",        # classical|chiller|sparse|soft
    siri_alpha=0.3,          # blend weight (soft mode only)
    siri_beta=5.0,           # sharpening factor (chiller mode)
)
output, attn_matrix = attn(x, return_attention=True)
```

### Invariantes formales (preservados del Bubble Transformer original)

- **Costo geométrico**: `C_ij = ‖Q_i - K_j‖²` (NO producto interno)
- **SIRI doubly-stochastic**: `A ∈ Σₙ` (politopo de Birkhoff)
- **Power Diagram ψ**: bias en log_Sinkhorn
- **ε bandwidth**: rango operativo [0.001, 1.0]
- **NumPy contract**: módulos core sin PyTorch
- **τ = 5 iteraciones**: Sinkhorn convergence

---

## 📁 Estructura

```
LLM-BUBBLE/
├── experiments/
│   ├── run_experiment.py          # Orchestrator principal
│   ├── config.py                  # Dataclass config + get_config()
│   ├── plateau_attention.py       # SIRI core (log-domain Sinkhorn) — KEEP
│   ├── deltanet_attention.py      # NEW: DeltaNet base attention
│   ├── siri_postprocess.py        # NEW: SIRI as opt-in post-processor
│   ├── power_diagrams.py          # NEW: ψ as explicit Laguerre bias
│   ├── hybrid_attention.py        # NEW: DeltaNet + SIRI + ψ combination
│   ├── metrics.py                 # 6 concentration/geometry metrics
│   ├── spectral_metrics.py        # SIGMA paper collapse detection
│   ├── epsilon_sweep.py           # Sweep controller + sweet spot
│   ├── visualize.py               # 7 plot generators
│   ├── extract_embeddings.py      # Qwen model extraction (GPU)
│   ├── generate_mock_embeddings.py # Synthetic embeddings
│   ├── tensor_compat.py           # NumPy fallback for PyTorch
│   └── v3_core.py, v4_adapter.py  # Legacy support (kept)
├── data/
├── embeddings/
├── results/
├── plots/
├── docs/
│   ├── decisions/
│   │   ├── 2026-06-27-sota-replacement-siri-preserved.md  # Architectural decision
│   │   └── 2026-06-27-siri-power-diagram-math.md          # Mathematical formalism
│   └── references.bib             # 17 papers BibTeX
├── tests/
│   ├── test_attention.py          # PlateauAttention (SIRI)
│   ├── test_metrics.py            # Concentration metrics
│   ├── test_power_diagrams.py     # NEW: Power Diagram ψ
│   ├── test_deltanet_attention.py # NEW: DeltaNet delta rule
│   ├── test_siri_postprocess.py   # NEW: SIRI post-processing
│   ├── test_hybrid_attention.py   # NEW: Hybrid DeltaNet + SIRI + ψ
│   └── ... (24 more test files)
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar experimento completo (mock mode, no GPU)
python experiments/run_experiment.py --mode mock

# Real mode (needs GPU + Qwen3-0.6B)
python experiments/run_experiment.py --mode real

# Tests
python -m pytest tests/ -v
```

---

## 📊 Métricas

| Métrica | Qué mide | Interpretación |
|---------|----------|----------------|
| **Effective Rank** | Dimensiones efectivas del embedding | Alto = expressivo, Bajo = colapsado |
| **Intrinsic Dim (MLE)** | Dimensión del manifold subyacente | Revela la verdadera complejidad |
| **Anisotropy Index** | Ratio eigenvalue máx/suma | 1.0 = colapso direccional |
| **Pairwise Distance Stats** | Distribución de distancias | Mean/std bajo = alta concentración |
| **Concentration Ratio** | Fracción activa en matriz de atención | Bajo = atención esparsa/concentrada |
| **Attention Entropy** | Entropía de la distribución de atención | Bajo = atención peaked |

---

## 🔬 Pipeline Experimental

```
┌─────────────────────────────────────────────────────┐
│                 Qwen 3.6 (Frozen)                    │
│  Input: corpus → extraer embeddings por capa         │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐             ┌────▼─────────┐
    │Softmax  │             │Hybrid         │
    │(baseline│             │DeltaNet + SIRI │
    │  Qwen)  │             │+ Power Diagram│
    └────┬────┘             └────┬─────────┘
         │                       │
         │              ┌────────┼────────┐
         │              │        │        │
         │           λ=1.0   λ=0.5   λ=0.0
         │           (Delta) (Hybrid) (SIRI)
         │              │        │        │
         └──────┬───────┴────────┴────────┘
                │
     ┌──────────▼──────────┐
     │  6 Métricas por capa │
     │  + Heatmaps + t-SNE  │
     │  + Pareto Frontier   │
     │  + Sweet Spot Report │
     └─────────────────────┘
```

## 🧬 SIRI-Soft Variants (NEW, June 2026)

The classical doubly-stochastic SIRI destroys attention peakedness. We identified 3 variants that preserve it:

| Variant | Formula | Use case | PPL (L3, λ=0.5) |
|---------|---------|----------|------------------|
| **Soft blend** | `(1-α)·softmax + α·SIRI` | Best balanced | **26.76** |
| Classical | Sinkhorn(-C/ε) | Strict doubly-stoch | 30.14 |
| Chiller (β) | Sinkhorn(scores·β) | Sharper peaks | 39.39 |
| Sparse (ReLU) | Sinkhorn(ReLU(-C/ε)) | Very sparse | — |

**Empirical evidence** (Qwen3-0.6B, layer 3 swap):
- Baseline: PPL 23.37
- Soft blend (α=0.7): PPL 26.76 (Δ +3.39)
- Pure SIRI: PPL 30.14 (Δ +6.77, ~2× worse)

See `results_real/PERPLEXITY_REPORT.md` and `experiments/siri_soft.py` for details.

---

## 🧪 Ejecutar Tests

```bash
# Suite completa
python -m pytest tests/ -v

# Test individual de cada módulo
python experiments/plateau_attention.py
python experiments/deltanet_attention.py
python experiments/power_diagrams.py
python experiments/siri_postprocess.py
python experiments/hybrid_attention.py

# Test específico
python -m pytest tests/test_hybrid_attention.py -v
```

**Estado actual**: 462 tests passing, 2 skipped, 0 failed (June 28, 2026).

---

## 📖 Referencia Teórica

### Bubble Transformer (TSM) — preservado

La atención se reformula como un problema de **Transporte Óptimo Entrópico**:

$$\mathcal{E}(A) = \langle A, C \rangle - \epsilon \cdot H(A)$$

Donde:
- $A$ = matriz de atención (superficie mínima)
- $C_{ij} = \|Q_i - K_j\|_2^2$ = matriz de costo geométrico (NO QK⊤)
- $H(A)$ = entropía de Shannon (presión interna)
- $\epsilon$ = bandwidth/temperatura (controla sparsity)

### SIRI (Sinkhorn Iterative Regularized Inference)

Algoritmo de Sinkhorn-Knopp en dominio logarítmico (preservado):

```
log_S = -C / ε + ψ  # log-domain + Power Diagram bias
u, v = 0             # potenciales duales
for τ iterations:
    u = -logsumexp(log_S + v, axis=-1)
    v = -logsumexp(log_S + u, axis=-2)
A = exp(log_S + u + v)  # doubly-stochastic
```

Convergencia: $\tau = 5$ iteraciones, error $O(\exp(-10\epsilon \sigma_{max}(C)))$.

### DeltaNet (NeurIPS 2024)

Para cada token $t$:
```
v_old = S_{t-1}^T k_t        # retrieve
delta = v_t - v_old          # correction
S_t = S_{t-1} + k_t delta^T   # update
o_t = S_t^T q_t              # output
```

O(N) en inference, paralelizable por chunks.

### ε como Bandwidth

- **ε → 0**: Atención colapsa a one-hot (máxima concentración)
- **ε → ∞**: Atención converge a uniforme (sin sparsity)
- **Sweet spot**: ε ≈ 0.001 (validado empíricamente para Qwen3)

---

## 📝 Decisión Arquitectónica

Ver [`docs/decisions/2026-06-27-sota-replacement-siri-preserved.md`](docs/decisions/2026-06-27-sota-replacement-siri-preserved.md) para análisis completo de 8 papers SOTA.

**Top-2 candidatas evaluadas**:
1. **DeltaNet** (Yang et al. 2024) — default, NeurIPS-grade, O(N)
2. **Kimi Linear / KDA** (Kimi Team 2025) — opt-in, Oct 2025, drop-in replacement

**Arquitectura adoptada**: Hybrid DeltaNet (default) + SIRI post-processing + Power Diagram ψ.

SDOT fue eliminado por decisión del usuario (junio 2026); SIRI y Power Diagram se conservan como invariantes formales.

---

## 📝 Notas de Implementación

- **Modelo**: Qwen 3.6 usa atención híbrida (3 capas DeltaNet + 1 full attention). Las capas objetivo son las de full attention: [3, 7, 11, 15, 19, 23]
- **Dominio logarítmico**: Sinkhorn previene underflow numérico en ε < 0.01
- **τ = 5 iteraciones**: Convergencia práctica (Sinkformers paper)
- **bfloat16**: Para eficiencia de memoria durante extracción
- **Power Diagram ψ**: bias aditivo en log_Sinkhorn, learnable projection W_ψ

---

## 🔮 Próximos Pasos

1. **Plan C**: Kimi Linear / KDA opt-in para máxima SOTA (2025)
2. **Plan D**: Layer selection adaptativo con HybridAttention
3. **Plan E**: Cost Matrix Engineering (mejores funciones de costo que L2)

---

*LLM-BUBBLE v0.2 · Bubble Transformer Research · Junio 2026*  
*Migración SDOT → DeltaNet completada · SIRI y Power Diagram preservados*
