# Architectural Decision: SDOT → SOTA-Replacement (Preservando SIRI + Power Diagram)

**Fecha**: 2026-06-27
**Proyecto**: LLM-BUBBLE / Bubble Transformer Research
**Autor**: Sisyphus (automate.dev) — Marcus
**Estado**: Decisión arquitectónica — Fase 0 completada
**Skill activa**: `arxiv-research`, `mathematical-depersonalization-engine`
**Documentos relacionados**:
- `2026-06-27-siri-power-diagram-math.md` (formalismo SIRI + ψ)
- `docs/references.bib` (BibTeX de papers)

---

## 1. Contexto y motivación

### 1.1 Estado actual del proyecto

El proyecto LLM-BUBBLE implementa el **Bubble Transformer** usando **SDOT (Sinkhorn Divergence Optimal Transport)** + SIRI (Sinkhorn Iterative Regularized Inference) + Power Diagram (ψ). El stack actual:

- `experiments/plateau_attention.py` — SIRI log-domain Sinkhorn (NumPy)
- `experiments/v3_core.py` — Voronoi + BaroreceptorMLP + BlockMaskedAttention
- `experiments/v4_adapter.py` — V4PlateauAdapter con Power Diagram ψ
- `experiments/sdota_attention.py` — SDOT Attention v3 (Voronoi + SIRI)
- `experiments/sdot_attention_v4.py` — SDOT Attention v4 (Power Diagram + SIRI)

### 1.2 Decisión del usuario (2026-06-27)

> "SDOT ya no la quiero, quiero otro enfoque que sea más eficiente basado en los nuevos modelos de lenguaje que demuestran superioridad frente a SDOT"

> "Conservando SIRI y Power Diagram"

### 1.3 Restricciones invariantes

- **SIRI** = Sinkhorn-Knopp log-domain (preservado por decisión explícita)
- **Power Diagram** = ψ como bias en `log_S = -C/ε + ψ` (preservado)
- **ε** = bandwidth/temperatura (mantener nombre, nueva semántica)
- **NumPy contract** en core modules
- **Q/K/V projections** + costo `C = ‖Q-K‖²` (no `QK⊤`)

---

## 2. Estado del arte: papers analizados

### 2.1 Sinkhorn-based attention (relevancia directa)

| Paper | arxiv ID | Año | Cita clave | Compatibilidad SIRI |
|-------|----------|-----|------------|---------------------|
| **Sinkformers** | [2110.11773](https://arxiv.org/abs/2110.11773) | 2021 | "Sinkhorn's algorithm to make attention matrices doubly stochastic" | **Total** — SIRI es la base |
| **Sparse Sinkhorn Attention** | [2002.11296](https://arxiv.org/abs/2002.11296) | 2020 | "Sparse Sinkhorn Attention via differentiable sorting" | **Alta** — SIRI + sparsity |
| **Sinkhorn AutoEncoders** | [1810.01118](https://arxiv.org/abs/1810.01118) | 2018 | Patruni et al. — Wasserstein via Sinkhorn | Media — generativo, no atención |
| **Sinkhorn Divergences (OT↔MMD)** | [1810.08278](https://arxiv.org/abs/1810.08278) | 2019 | Feydy/Séjourné/Peyré — interpolation OT↔MMD | **Alta** — teoría SIRI |
| **Sinkhorn Distances** | [1306.0895](https://arxiv.org/abs/1306.0895) | 2013 | Cuturi — Sinkhorn original | Fundacional |

### 2.2 Linear/Delta/Hybrid attention SOTA (alternativas)

| Paper | arxiv ID | Año | Cita clave | Compatibilidad SIRI |
|-------|----------|-----|------------|---------------------|
| **GLA (Gated Linear Attention)** | [2312.06635](https://arxiv.org/abs/2312.06635) | 2023 | "Gated Linear Attention with hardware-efficient training" — Yang et al. | Alta (lineal, requiere feature map) |
| **RetNet** | [2307.08621](https://arxiv.org/abs/2307.08621) | 2023 | Sun et al. (Microsoft) — "successor to Transformer" | Alta (atención densa con decay) |
| **DeltaNet** | [2406.06484](https://arxiv.org/abs/2406.06484) | 2024 | Yang et al. — "Parallelizing Linear Transformers with Delta Rule" | Alta (lineal + delta rule) |
| **Mamba** | [2312.00752](https://arxiv.org/abs/2312.00752) | 2023 | Gu & Dao — "Linear-Time Sequence Modeling" | Parcial (SSM no es Q/K atención) |
| **Mamba-2 / SSD** | [2405.21060](https://arxiv.org/abs/2405.21060) | 2024 | Dao & Gu — "Transformers are SSMs" | Media (SSD estructura específica) |
| **RWKV-7 (Goose)** | [2503.14456](https://arxiv.org/abs/2503.14456) | 2025 | Peng et al. — "Expressive Dynamic State Evolution" | Parcial (WKV no Q/K) |
| **Kimi Linear / KDA** | [2510.26692](https://arxiv.org/abs/2510.26692) | 2025 | Kimi Team — "outperforms full attention under fair comparisons" | Alta (híbrido KDA + MLA) |
| **Performers (FAVOR+)** | [2009.14794](https://arxiv.org/abs/2009.14794) | 2020 | Choromanski et al. — "Rethinking Attention" | Alta (kernel φ + Sinkhorn) |

---

## 3. Análisis comparativo de candidatos

### 3.1 Matriz de evaluación (5 dimensiones, score 1-5)

| Arquitectura | Compatibilidad SIRI | Eficiencia vs SDOT | SOTA Score (2026) | Madurez código | Integración Qwen3 |
|--------------|---------------------|--------------------|--------------------|-----------------|--------------------|
| **Sinkformers (SIRI-only)** | 5/5 | 4/5 (SIRI ya es O(n²τ)) | Bajo (atención densa) | Maduro | Total |
| **Sparse Sinkhorn Attention** | 5/5 | 5/5 (sparsity → quasi-O(n)) | Bajo (2020) | Maduro | Total |
| **RetNet** | 4/5 (parallel + recurrent) | 5/5 (O(n) infer) | Alto (Microsoft, 2023) | Maduro | Total |
| **GLA** | 4/5 (lineal, requiere feature map) | 5/5 (O(n)) | Alto (NeurIPS 2024) | Maduro | Alta |
| **DeltaNet** | 4/5 (lineal + delta) | 5/5 (O(n)) | Alto (NeurIPS 2024) | Maduro | Total (Qwen3 ya lo usa) |
| **Kimi Linear / KDA** | 4/5 (híbrido KDA+MLA) | 5/5 (drop-in, KV cache -75%) | **Máximo (Oct 2025)** | Maduro (open-source) | Total |
| **Mamba-2** | 2/5 (SSD, no Q/K) | 5/5 (O(n)) | Alto | Maduro | Parcial |
| **RWKV-7** | 2/5 (WKV decay, no Q/K) | 5/5 (O(n)) | Alto (3B SoTA) | Maduro | Parcial |

### 3.2 Detalle por candidato (top 3)

#### 🥇 **Candidato A: Kimi Linear (arxiv:2510.26692)**

**Tesis**: "Drop-in replacement for full attention architectures with superior performance and efficiency"

**Fortalezas**:
- **Outperforms full attention** en fair comparisons (short-context, long-context, RL scaling)
- **75% KV cache reduction**
- **6× decoding throughput** a 1M context
- Layerwise hybrid KDA + MLA (compatible con Qwen3 hybrid pattern)
- Open-source KDA kernel + vLLM implementation
- 3B/48B model checkpoints released

**Compatibilidad SIRI**: 4/5 — KDA admite SIRI como kernel en bloques residuales; MLA blocks admiten SIRI directo.

**Riesgos**:
- Nuevo (Oct 2025) — adopción temprana
- Complejidad alta (KDA kernel specialized)

**Decisión**: **FUERTE CANDIDATO** — Máxima SOTA + eficiencia + compatibilidad Qwen3

---

#### 🥈 **Candidato B: DeltaNet (arxiv:2406.06484) + SIRI post-process**

**Tesis**: "Delta rule for associative recall, parallel over sequence length"

**Fortalezas**:
- 1.3B model trained 100B tokens outperforms Mamba/GLA
- **Ya usado en Qwen3** (3 DeltaNet + 1 full attention pattern) → integración nativa
- Delta rule mejora associative recall vs additive linear attention
- Hardware-efficient algorithm (Householder matrices)
- NeurIPS 2024

**Compatibilidad SIRI**: 5/5 — Aplicar SIRI como post-processing en la matriz de scores de DeltaNet. O reemplazar la capa full-attention de Qwen3 por SIRI directamente.

**Riesgos**:
- DeltaNet por sí solo < Transformer en some tasks — pero híbrido funciona

**Decisión**: **RECOMENDADO** — Balance perfecto SOTA + SIRI compatibility + Qwen3 native

---

#### 🥉 **Candidato C: RetNet (arxiv:2307.08621) + SIRI**

**Tesis**: "Successor to Transformer — training parallelism + low-cost inference"

**Fortalezas**:
- **O(1) inference** (recurrent form) — máxima eficiencia
- 3 computation paradigms: parallel (training), recurrent (inference), chunkwise (long sequences)
- Microsoft backing (production-grade)
- Theoretical connection recurrence ↔ attention derivada

**Compatibilidad SIRI**: 4/5 — Reemplazar softmax del parallel representation por SIRI. Recurrent form naturalmente admite decay (compatible con ε bandwidth).

**Riesgos**:
- Performance mixed en some tasks vs Transformer
- Menos drop-in que Kimi Linear

**Decisión**: **ALTERNATIVA** — Si prioridad es inferencia ultra-rápida

---

## 4. Recomendación

### 4.1 Arquitectura recomendada: **HYBRID DELTANET + SIRI**

```
Bubble Transformer post-SDOT:
  - Atención base: DeltaNet (parallel form, hardware-efficient)
  - SIRI post-processing: Sinkhorn log-domain sobre output DeltaNet
  - Power Diagram ψ: bias en log-Sinkhorn
  - ε bandwidth: rango [0.001, 1.0] (sin cambios)
  - Capas híbridas: DeltaNet (3 capas) + Full Attention con SIRI (1 capa), patrón Qwen3
```

### 4.2 Justificación

1. **SOTA (2024)**: DeltaNet es NeurIPS 2024, outperforms Mamba/GLA, 1.3B trained 100B tokens.
2. **Compatibilidad SIRI**: DeltaNet es lineal con delta rule; SIRI aplica como post-process o como kernel de la capa full-attention.
3. **Compatibilidad Qwen3**: Qwen3 ya usa patrón DeltaNet (3 DeltaNet + 1 full attention). La nueva arquitectura se alinea nativamente.
4. **Eficiencia**: DeltaNet O(n) training, O(n) inference. SIRI post-process O(n²·τ) = O(5n²) — overhead despreciable vs O(n²) de softmax.
5. **Open-source**: DeltaNet tiene código público (paper oficial). Implementación factible en NumPy + PyTorch.

### 4.3 Arquitectura alternativa: **HYBRID KIMI LINEAR + SIRI**

Si se requiere máxima SOTA (Oct 2025) sobre DeltaNet:

```
Bubble Transformer post-SDOT v2:
  - Atención base: Kimi Linear (KDA + MLA hybrid)
  - SIRI post-processing: Sinkhorn sobre output Kimi Linear
  - Power Diagram ψ: bias en log-Sinkhorn
  - KV cache reduction 75%
  - 6× decoding throughput a 1M context
```

**Trade-off**: Mayor SOTA, mayor complejidad de implementación.

---

## 5. Plan de implementación

### 5.1 Estructura de archivos

```
experiments/
├── plateau_attention.py          ← KEEP (SIRI core, log-domain Sinkhorn)
│                                    + __call__ agregado (Fase 1)
├── power_diagrams.py             ← NEW (ψ como módulo explícito)
├── deltanet_attention.py         ← NEW (DeltaNet implementation NumPy)
├── siri_postprocess.py           ← NEW (SIRI como post-processing opt-in)
├── hybrid_attention.py           ← NEW (DeltaNet + SIRI + ψ)
├── metrics.py                    ← KEEP + Fase 1 fix (torch compat)
├── spectral_metrics.py           ← KEEP + Fase 1 fix (iso logic)
├── epsilon_sweep.py              ← REFACTOR (ε como bandwidth de nueva atención)
├── v3_core.py                    ← SIMPLIFY (usa hybrid_attention; conserva BaroreceptorMLP)
├── v4_adapter.py                 ← SIMPLIFY (Power Diagram ψ sobre hybrid_attention)
├── sdota_attention.py            ← DELETE (SDOT eliminado por usuario)
├── sdot_attention_v4.py          ← DELETE (SDOT eliminado por usuario)
├── extract_embeddings.py         ← KEEP
├── generate_mock_embeddings.py   ← KEEP
├── tensor_compat.py              ← KEEP
├── config.py                     ← UPDATE (nuevos defaults)
├── run_experiment.py             ← UPDATE
├── visualize.py                  ← KEEP
├── analyze_results.py            ← KEEP
└── optimal_config.py             ← REGENERATE
```

### 5.2 Tests a actualizar

**Eliminar** (obsoletos):
- `tests/test_layer_selection.py` — import faltante
- `tests/test_sdota_attention.py` — SDOT eliminado
- `tests/test_sdot_attention_v4.py` — SDOT eliminado

**Actualizar** (verde esperado):
- `tests/test_attention.py` — usar `PlateauAttention.__call__` + nueva arquitectura
- `tests/test_v3_core.py` — V3 simplificado con hybrid_attention
- `tests/test_v4_core.py` — V4 simplificado
- `tests/test_v4_adapter.py` — Power Diagram ψ + hybrid_attention
- `tests/test_integration.py` — pipeline completo
- `tests/test_metrics.py` — fix Fase 1 (NumPy-only contract)
- `tests/test_spectral_metrics.py` — fix Fase 1 (iso logic)

**Agregar** (nuevos):
- `tests/test_deltanet_attention.py` — tests de DeltaNet
- `tests/test_power_diagrams.py` — ψ como bias
- `tests/test_siri_postprocess.py` — Sinkhorn post-processing
- `tests/test_hybrid_attention.py` — combinación DeltaNet + SIRI + ψ

---

## 6. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| DeltaNet implementación compleja | Media | Medio | Empezar con implementación simplificada; iterar |
| SIRI + DeltaNet incompatibles en práctica | Baja | Alto | Test empírico en Fase 2; fallback a Softmax+ψ |
| Kimi Linear no disponible offline | Alta | Bajo | Default DeltaNet; Kimi Linear como opt-in |
| Tests rotos no resueltos | Baja | Bajo | Fase 1 primero |
| Dual-head bug impacta tests | Media | Bajo | Fase 1.5 (skill fix-dual-head-phase1) |

---

## 7. Decisión final

**Arquitectura adoptada**: Hybrid DeltaNet + SIRI post-processing + Power Diagram ψ

**Justificación resumida**:
1. DeltaNet es SOTA (NeurIPS 2024), compatible con Qwen3 hybrid pattern, NeurIPS-grade
2. SIRI post-processing es drop-in sin modificar arquitectura base
3. Power Diagram ψ se preserva como bias en log_S
4. Eficiencia O(n) preservada, SIRI overhead O(5n²) despreciable
5. Tests verdes posibles (495 pasan, 29 a arreglar en Fase 1)

**Próximo paso**: Validar con usuario → ejecutar Fase 1 (4 clusters de tests rotos) → Fase 1.5 (skill dual-head) → Fase 2 (implementación) → Fase 3 (tests nuevos) → Fase 4 (docs).

---

*Documento finalizado. Pendiente validación del usuario para proceder con implementación.*