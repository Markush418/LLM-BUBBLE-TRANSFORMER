# Test Reales — LLM-BUBBLE v4

**Fecha**: 16 Abril 2026
**Modelo**: Qwen3-0.6B-Base (4-bit NF4 quantized)
**Hardware**: NVIDIA GTX 1650 (4GB VRAM) + CUDA 12.8

---

## Resumen Ejecutivo

Se ejecutó exitosamente el experimento Bubble Transformer V4 con embeddings reales extraídos de Qwen3-0.6B-Base. El **sweet spot encontrado es ε = 0.001**, logrando:

- **Effective Rank**: 606.7 (preserva expressividad sin colapso)
- **Concentration Ratio**: 0.044 (atención altamente concentrada)
- **Anisotropy Index**: 0.027 (distribución direccional balanceada)

---

## Configuración del Experimento

| Parámetro | Valor |
|-----------|-------|
| Modelo | Qwen3-0.6B-Base |
| Cuantización | 4-bit NF4 (double quantization) |
| Capas objetivo | [0, 4, 8, 12, 16, 20, 24] |
| d_model | 1024 |
| num_heads | 16 |
| head_dim | 64 |
| τ (Sinkhorn iterations) | 5 |
| Seed | 42 |

---

## Archivos de Documentación

1. **[01-configuracion.md](./01-configuracion.md)** — Setup del environment y parámetros
2. **[02-extraccion-embeddings.md](./02-extraccion-embeddings.md)** — Proceso de extracción y problemas encontrados
3. **[03-epsilon-sweep.md](./03-epsilon-sweep.md)** — Resultados del sweep de epsilon
4. **[04-tension-sweep.md](./04-tension-sweep.md)** — Resultados del sweep dual-head
5. **[05-metricas-finales.md](./05-metricas-finales.md)** — Métricas finales y análisis
6. **[06-visualizaciones.md](./06-visualizaciones.md)** — Descripción de plots generados
7. **[07-problemas-soluciones.md](./07-problemas-soluciones.md)** — Problemas encontrados y soluciones aplicadas

---

## Resultados Principales

### Sweet Spot: ε = 0.001

```
┌─────────────────────────────────────────────────────────┐
│  ε=0.001: Máxima concentración con preservación de     │
│  dimensionalidad efectiva (606.7 vs baseline 624)      │
│                                                         │
│  Trade-off óptimo entre:                                │
│  - Atención concentrada (conc_ratio = 0.044)           │
│  - Expressividad preservada (eff_rank = 606.7)         │
│  - Sin colapso representacional                         │
└─────────────────────────────────────────────────────────┘
```

### Comparativa por Epsilon

| ε | Effective Rank | Concentration | Anisotropy | Estado |
|---|----------------|---------------|------------|--------|
| 0.001 | 606.7 | 0.044 | 0.027 | **ÓPTIMO** |
| 0.01 | 580.7 | 0.271 | 0.020 | OK |
| 0.1 | 355.0 | 0.487 | 0.049 | Alta concentración |
| 1.0 | 329.2 | 0.500 | 0.050 | Casi uniforme |

---

## Capas Analizadas

```
Layer 0 (embedding):  eff_rank = 624.0 (baseline)
Layer 4 (early):      eff_rank = 603.9 (compresión moderada)
Layer 8 (early-mid):  eff_rank = 618.3 
Layer 12 (mid):       eff_rank = 582.7 (máxima compresión)
Layer 16 (mid-late):  eff_rank = 608.7
Layer 20 (late):      eff_rank = 666.4 (expansión)
Layer 24 (final):     eff_rank = 694.7 (máxima expressividad)
```

**Patrón observado**: Las capas intermedias (8-16) comprimen más, las capas finales (20-24) expanden.

---

## Tiempo de Ejecución

| Fase | Tiempo |
|------|--------|
| Extracción de embeddings | 32s |
| Epsilon sweep (20 combinaciones) | 3m 27s |
| Tension sweep (24 combinaciones) | 6m 21s |
| Generación de plots | <10s |
| **Total** | **~10 minutos** |

---

## Comandos Ejecutados

```bash
# 1. Extracción de embeddings
python experiments/extract_embeddings.py --target-layers 0 4 8 12 16 20 24 --batch-size 1 --max-length 256

# 2. Clipping de outliers
python -c "..."  # Ver 02-extraccion-embeddings.md

# 3. Epsilon sweep
python experiments/run_experiment.py --mode real --d-model 1024 --num-heads 16 \
    --epsilon-values 0.001 0.01 0.1 1.0 \
    --target-layers 0 4 12 20 24 --skip-generation

# 4. Tension sweep
python experiments/run_experiment.py --mode tension --d-model 1024 --num-heads 16 \
    --epsilon-values 0.001 0.01 0.1 \
    --target-layers 0 4 12 24 --skip-generation
```

---

## Próximos Pasos

1. **Validación cruzada**: Ejecutar con corpus diferente
2. **Análisis de sensibilidad**: Probar τ = 3, 7, 10
3. **Comparación con baseline**: Softmax vs Plateau lado a lado
4. **Aplicación práctica**: Integrar en modelo fine-tuned

---

*Documentación generada automáticamente desde los logs de ejecución.*
*LLM-BUBBLE v4 — Bubble Transformer Research*
