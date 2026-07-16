# Bubble Transformer

> **Atención híbrida con transporte óptimo entrópico** · Investigación independiente de [kyan-labs](https://kyan-labs.com)

<!-- BADGES: actualizar el conteo de tests si cambia, agregar el DOI de Zenodo una vez enviado -->
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-475%2F490_passing-brightgreen.svg)
<!-- TODO: badge de DOI de Zenodo -->

*Esta es la traducción al español de [README.md](README.md), que es la versión canónica. Ante cualquier discrepancia, el README en inglés manda.*

---

## TL;DR

Bubble Transformer reemplaza la atención softmax por una **formulación de transporte óptimo entrópico** (SIRI) combinada con el **recall asociativo O(N) de DeltaNet**. En Qwen3-0.6B observamos **SIRI — Sparsity-Induced Rank Inflation** — un fenómeno empírico no monótono donde el rango efectivo de atención alcanza un pico de **2.89× el baseline de softmax** en un ancho de banda ε específico, y que está ausente en matrices aleatorias.

**Estado actual (V5)**: arquitectura híbrida (DeltaNet + SIRI + Power Diagram ψ) validada en Qwen3-0.6B · 475/490 tests pasando en un entorno CI limpio, 489/490 en la máquina de referencia con GPU · gate de ΔPPL en swap de una sola capa: PASS, con **FocusDeltaNet en la capa 7 (λ=0.3) alcanzando +0.16% vs el baseline de softmax** — 12.5× por debajo del umbral del 2%.

---

## Hallazgo clave — SIRI (Sparsity-Induced Rank Inflation)

La atención normalizada con Sinkhorn exhibe un fenómeno de inflación de rango ausente tanto en softmax como en baselines aleatorios.

**Medición en Qwen3-0.6B**:

| Tipo de atención           | Rango efectivo (R_eff) |
| --------------------------- | ---------------------- |
| Matriz aleatoria (control)  | ~1.0                   |
| Baseline softmax            | 199.6                  |
| **SIRI @ ε=0.005 (pico)**   | **576.5 (2.89× softmax)** |

R_eff varía de forma no monótona con ε. La ubicación del pico es estable entre semillas. Como no aparece en matrices aleatorias, el efecto es una propiedad de la normalización de Sinkhorn interactuando con la geometría de atención aprendida — no un artefacto numérico.

<!-- TODO: incrustar el gráfico R_eff vs ε desde assets/ (Captura.PNG o uno nuevo) -->

---

## Arquitectura

```
                   Embeddings de entrada
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
        Ruta DeltaNet                Ruta SIRI
     (recall lineal O(N))     (Sinkhorn log-domain
                                + Power Diagram ψ)
               │                           │
               └─────────────┬─────────────┘
                             ▼
                 Interpolación híbrida
                 out = λ · delta + (1−λ) · siri
```

Tres componentes:

- **DeltaNet** (Yang et al. 2024) — atención de tiempo lineal con delta rule para recall asociativo
- **SIRI post-processing** — normalización doblemente estocástica de Sinkhorn–Knopp en dominio logarítmico, τ = 5 iteraciones
- **Power Diagram ψ** — bias de teselación de Laguerre aprendible en `log_S = −C/ε + ψ`

La matriz de costo usa distancia geométrica L2: `C_ij = ‖Q_i − K_j‖²`, **no** el producto interno estándar `QK⊤`.

---

## Inicio rápido

```bash
pip install -r requirements.txt
```

```python
from hybrid_attention import HybridAttention

attn = HybridAttention(
    d_model=1024, num_heads=16,       # dims de Qwen3-0.6B
    epsilon=0.005,                    # bandwidth de SIRI en el pico de R_eff
    lam=0.5,                          # balance híbrido (1.0=DeltaNet, 0.0=SIRI)
    siri_mode="soft",                 # classical | chiller | sparse | soft
    siri_alpha=0.3,                   # peso de mezcla (modo soft)
)
output, attn_matrix = attn(x, return_attention=True)
```

---

## Resultados

### Resultado principal — swap de una capa en Qwen3-0.6B (WikiText-2, N=256, semilla 42)

Capa 7 de atención completa reemplazada, resto del modelo congelado. Reproducido el 2026-07-16 en una GTX 1650.

| Config                                | ε     | τ | λ   | PPL        | ΔPPL vs softmax |
| -------------------------------------- | ----- | - | --- | ---------- | --------------- |
| Baseline softmax                       | —     | — | —   | 22.513     | —               |
| Bubble Transformer @ L7 (solo Focus)  | 0.001 | 1 | 0.0 | 22.681     | +0.74% ✓        |
| **FocusDeltaNet @ L7 (V5)**            | 0.001 | 1 | 0.3 | **22.550** | **+0.16% ✓**    |

Criterio del gate: ΔPPL ≤ 2%. **V5 PASA 12.5× por debajo del umbral** — la primera configuración de BT V5 en lograrlo, después de 38 intentos fallidos previos.

Escalado multi-capa (FocusDeltaNet): {L7} → 22.550 (+0.16%), {L7, L10} → 22.648 (+0.60%), {L7, L10, L12} → 22.825 (+1.38%). Las tres configuraciones pasan el gate.

### Ablación — variantes SIRI-Soft en la capa 3 (λ = 0.5)

SIRI clásico (doblemente estocástico) destruye la nitidez ("peakedness") de la atención. Tres variantes la preservan en distinto grado:

| Variante                | Fórmula                     | PPL       | ΔPPL     |
| ------------------------ | --------------------------- | --------- | -------- |
| Baseline softmax         | —                            | 23.37     | —        |
| **Soft blend (α=0.7)**  | (1−α)·softmax + α·SIRI      | **26.76** | +14.5%   |
| SIRI clásico             | Sinkhorn(−C/ε)               | 30.14     | +29.0%   |
| Chiller (β=5)             | Sinkhorn(scores·β)          | 39.39     | +68.5%   |

Soft blend recupera aproximadamente la mitad de la degradación de SIRI clásico en la capa 3. Esta ablación fue lo que motivó el cambio a V5: en lugar de forzar una distribución doblemente estocástica mediante mezcla suave, usar Sinkhorn *solo* para agrupar tokens y dejar que softmax maneje la atención real dentro de cada grupo. Ese cambio es lo que cierra la brecha en L7 de +0.74% a +0.16%.

### Suite de tests

- CI limpio (sin GPU, sin caché de Qwen3): **475 pasando · 15 saltados · 0 fallados** (490 recolectados)
- Máquina de referencia con GPU (`RUN_QWEN3_TESTS=1`): **489 pasando · 1 saltado · 0 fallados**

Los tests de wrapper opt-in (`test_focus_bubble_wrapper.py`, `test_qwen3_hybrid_wrapper.py`) descargan Qwen3-0.6B, por lo que están controlados por `RUN_QWEN3_TESTS=1` para mantener `pytest tests/` reproducible por defecto en cualquier entorno.

---

## Método

La atención se formula como transporte óptimo entrópico:

$$
\mathcal{E}(A) = \langle A, C \rangle - \epsilon \cdot H(A)
$$

donde $A$ es la matriz de atención, $C_{ij} = \|Q_i - K_j\|_2^2$ es el costo geométrico, $H(A)$ es la entropía de Shannon, y $\epsilon$ es el ancho de banda (temperatura).

Iteración de Sinkhorn en dominio logarítmico (estabilidad numérica en ε < 0.01):

```
log_S = -C / ε + ψ                 # costo + bias de Power Diagram
u, v = 0, 0                        # potenciales duales
for τ in range(5):
    u = -logsumexp(log_S + v, axis=-1)
    v = -logsumexp(log_S + u, axis=-2)
A = exp(log_S + u + v)             # doblemente estocástica
```

Cota del error de convergencia: $O(\exp(-10 \epsilon \sigma_{\max}(C)))$.

Formalismo matemático completo en [`docs/decisions/2026-06-27-siri-power-diagram-math.md`](docs/decisions/2026-06-27-siri-power-diagram-math.md).

---

## Reproducibilidad

- **Modelo**: Qwen3-0.6B-Base, float16, atención eager. Arquitectura híbrida: 3 capas DeltaNet + 1 de atención completa, repetido. Índices de capas de atención completa: `[3, 7, 11, 15, 19, 23]`
- **Datos**: split de test de WikiText-2, 50k caracteres, N=256 tokens por ventana
- **Precisión**: bfloat16 durante la extracción de embeddings; float16 durante la evaluación de PPL
- **Iteraciones de Sinkhorn**: τ = 5 para SIRI legacy; τ = 1 para Focus Bubble V5 (softmax dentro de los grupos se encarga de la nitidez)
- **Semillas**: fijas en 42 en todos los experimentos reportados (ver `experiments/config.py`)
- **Hardware**: NVIDIA GTX 1650 (4.3 GB VRAM); ~30 minutos para la suite completa de benchmarks V5

Reproducir el resultado de V5 que pasa el gate:

```bash
python experiments/benchmark_focus_deltanet_sweep.py       # sweep de λ en L7 → mejor config
python experiments/benchmark_focus_layer_sweep_optimal.py  # sweep de capas en el óptimo
python experiments/benchmark_focus_fine_sweep.py            # sweep de ε, τ en L12
python experiments/benchmark_focus_multilayer.py            # escalado multi-capa
python experiments/niah_focus_bubble.py                     # Needle-in-a-Haystack a 2K
```

Experimentos legacy de V4 (siguen funcionando):

```bash
python experiments/run_experiment.py --mode real   # requiere GPU + Qwen3-0.6B
python experiments/run_experiment.py --mode mock   # solo CPU, embeddings sintéticos
```

Tests:

```bash
python -m pytest tests/ -v                         # 475 pasan, 15 saltados (opt-in)
RUN_QWEN3_TESTS=1 python -m pytest tests/ -v       # 489 pasan con tests reales de Qwen3
```

---

## Trabajo relacionado

- **DeltaNet** (Yang et al., NeurIPS 2024) — atención lineal con delta rule · [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
- **Sinkformers** (Sander et al., ICML 2022) — primera formulación de atención basada en Sinkhorn
- **Focus** (arXiv:2604.03260) — Sinkhorn para agrupar tokens con softmax adentro — inspiración directa para V5
- **Litman** (2025, arXiv:2508.08369) — SDPA como transporte óptimo entrópico one-sided (exacto) — fundamento teórico
- **LOTFormer** (arXiv:2509.23436) — atención doblemente estocástica en tiempo lineal — el competidor de V5 contra el cual comparar
- **Kimi Linear / KDA** (Kimi Team, 2025) — atención lineal SOTA (evaluada como alternativa opt-in)
- **SIGMA** (2024) — métricas de detección de colapso espectral usadas acá como diagnóstico

Bibliografía completa (17 papers) en [`docs/references.bib`](docs/references.bib). Fundamento arquitectónico de V5 en [`IMPORTANTE/BT-V5_06_focus_bubble.md`](IMPORTANTE/BT-V5_06_focus_bubble.md) y el borrador de arXiv en [`paper/main.tex`](paper/main.tex).

---

## Cita

Si usás Bubble Transformer o el hallazgo de SIRI en tu investigación, por favor citá:

```bibtex
@misc{bubble_transformer_2026,
  title        = {Bubble Transformer: Hybrid Attention with Entropic Optimal Transport},
  author       = {Marcus and kyan-labs},
  year         = {2026},
  howpublished = {\url{https://github.com/Markush418/LLM-BUBBLE-TRANSFORMER}},
  note         = {Independent research. Zenodo DOI: TODO}
}
```

<!-- TODO: completar el nombre completo del autor y el DOI de Zenodo una vez publicado -->

---

## Acerca de

**Bubble Transformer** es investigación independiente de **[kyan-labs](https://kyan-labs.com)** — un estudio independiente de investigación e ingeniería liderado por Marcus, con sede en Argentina.

kyan-labs brinda consultoría en:

- Optimización de costos de inferencia de LLMs
- Mecanismos de atención personalizados y arquitecturas de contexto largo
- Sistemas de orquestación multi-agente
- Ingeniería de compiladores para compresión semántica / de prompts

**Consultas de consultoría**: <!-- TODO: completar el email de contacto --> · [kyan-labs.com](https://kyan-labs.com)

---

## Licencia

MIT — ver [LICENSE](LICENSE)
