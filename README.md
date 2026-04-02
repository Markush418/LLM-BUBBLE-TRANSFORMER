# LLM-BUBBLE — Bubble Transformer Research

> **Embedding Geometry Analysis + ε Sweet Spot Discovery**  
> Plan A+B: Mapear distribución de embeddings bajo Sinkhorn vs Softmax y encontrar el coeficiente de viscosidad óptimo.

---

## 🎯 Objetivo

Determinar **qué embeddings** de Qwen 3.6 se concentran mejor en la "sección de gravedad" del Bubble Transformer, y **qué valor de ε** (viscosidad) maximiza esta concentración sin colapso representacional.

### Plan A — Embedding Geometry Map
Comparar cómo se distribuyen los embeddings de Qwen 3.6 bajo:
- **Softmax** (baseline, atención original de Qwen)
- **Plateau/Sinkhorn** (atención de superficie mínima, con ε variable)

### Plan B — ε Sweet Spot
Encontrar el rango de ε que produce máxima concentración de embeddings sin colapso dimensional.

---

## 📁 Estructura

```
LLM-BUBBLE/
├── experiments/
│   ├── extract_embeddings.py   # Extrae embeddings de Qwen 3.6 por capa
│   ├── plateau_attention.py    # PlateauAttentionMechanism (Sinkhorn-Knopp)
│   ├── epsilon_sweep.py        # Sweep de ε + análisis de sweet spot
│   ├── metrics.py              # Motor de 5 métricas de concentración
│   ├── visualize.py            # Generador de 7 tipos de plots
│   └── run_experiment.py       # Orchestrator principal (un comando)
├── data/
│   └── test_corpus.jsonl       # Corpus de prueba (auto-generado)
├── embeddings/
│   ├── softmax/                # Embeddings baseline de Qwen
│   └── plateau/                # Embeddings con PlateauAttention
├── results/
│   ├── epsilon_sweep.json      # Resultados completos del sweep
│   └── sweet_spot_analysis.md  # Reporte de recomendación
├── plots/
│   ├── effective_rank_curves.png
│   ├── concentration_heatmap_*.png
│   ├── pareto_frontier.png
│   ├── anisotropy_vs_epsilon.png
│   ├── intrinsic_dim_vs_epsilon.png
│   ├── tsne_layer_*.png
│   └── summary_dashboard.png
├── docs/superpowers/specs/
│   └── 2026-04-01-bubble-embedding-geometry-design.md
├── requirements.txt
├── texto.txt                   # Formalización matemática original TSM
├── readme.rtf                  # Documentación arquitectónica RTF
└── pyth.txt                    # Implementación PyTorch de referencia
```

---

## 🚀 Quick Start

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar experimento completo
python experiments/run_experiment.py --model "Qwen/Qwen3.6-Plus" --device cuda

# 3. Ver resultados
cat results/sweet_spot_analysis.md
ls plots/
```

### Opciones del Orchestrator

```bash
# Con corpus personalizado
python experiments/run_experiment.py --corpus data/my_corpus.jsonl

# Saltar extracción (si ya existen embeddings)
python experiments/run_experiment.py --skip-extraction

# Solo métricas, sin visualizaciones
python experiments/run_experiment.py --skip-visualization

# Valores de ε custom
python experiments/run_experiment.py --epsilon-values 0.01 0.05 0.1 0.5

# Capas objetivo custom
python experiments/run_experiment.py --target-layers 3 7 11 15
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
    │  5 Métricas por capa │
    │  + Heatmaps + t-SNE  │
    │  + Pareto Frontier   │
    │  + Sweet Spot Report │
    └─────────────────────┘
```

---

## 🧪 Ejecutar Tests Unitarios

```bash
# Test individual de cada módulo
python experiments/plateau_attention.py   # Test PlateauAttention
python experiments/metrics.py             # Test métricas

# Test completo del pipeline (sin modelo real)
python experiments/test_pipeline.py       # Mock-based validation
```

---

## 📖 Referencia Teórica

### Bubble Transformer (TSM)
La atención se reformula como un problema de **Transporte Óptimo Entrópico**:

$$\mathcal{E}(A) = \langle A, C \rangle - \epsilon H(A)$$

Donde:
- $A$ = matriz de atención (superficie mínima)
- $C_{ij} = \|Q_i - K_j\|_2^2$ = matriz de costo geométrico
- $H(A)$ = entropía de Shannon (presión interna)
- $\epsilon$ = coeficiente de viscosidad (controla sparsity)

### Algoritmo de Sinkhorn-Knopp
Resuelve la minimización en dominio logarítmico:

```
log_S = -C / ε
u, v = 0  # potenciales duales
for τ iterations:
    u = -logsumexp(log_S + v)
    v = -logsumexp(log_S + u)
A = exp(log_S + u + v)
```

### ε y Concentración
- **ε → 0**: Atención colapsa a one-hot (máxima concentración, pérdida de expressividad)
- **ε → ∞**: Atención converge a uniforme (mínima concentración, sin sparsity)
- **Sweet spot**: Máxima concentración con effective rank ≥ 50% del baseline

---

## 📝 Notas de Implementación

- **Modelo**: Qwen 3.6 usa atención híbrida (3 capas DeltaNet + 1 full attention). Las capas objetivo son las de full attention: [3, 7, 11, 15, 19, 23]
- **Dominio logarítmico**: Previene underflow numérico en Sinkhorn
- **τ = 5 iteraciones**: Suficiente para convergencia práctica (verificado en Sinkformers paper)
- **bfloat16**: Para eficiencia de memoria durante extracción de embeddings

---

## 🔮 Próximos Pasos (Post Plan A+B)

1. **Plan C**: Dual-Head Tension — prevenir colapso representacional
2. **Plan D**: Layer Selection — qué capas de Qwen reemplazar
3. **Plan E**: Cost Matrix Engineering — mejores funciones de costo que L2

---

*LLM-BUBBLE v0.1 · Bubble Transformer Research · Abril 2026*
