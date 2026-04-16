# 04. Tension Sweep (Dual-Head)

**Fecha**: 16 Abril 2026
**Tiempo**: 6 minutos 21 segundos

---

## Comando Ejecutado

```bash
python experiments/run_experiment.py --mode tension \
    --d-model 1024 --num-heads 16 \
    --epsilon-values 0.001 0.01 0.1 \
    --target-layers 0 4 12 24 \
    --skip-generation
```

---

## Concepto: Dual-Head Tension

### Motivación

El mecanismo de **dual-head tension** busca balancear:

1. **Cabeza de concentración** (ε_low = 0.001): Maximiza la concentración de atención
2. **Cabeza de expressividad** (ε_high = 0.1): Preserva dimensionalidad

La **tensión** entre ambas se controla con el parámetro α:

```
output = α * head_low + (1 - α) * head_high

α = 0.0:  Output = head_high      (máxima expressividad)
α = 0.5:  Output = 50/50          (balance)
α = 1.0:  Output = head_low       (máxima concentración)
```

---

## Output Completo

```
[Step 2b/4] Running tension sweep (dual-head comparison)...
--------------------------------------------------
[Tension] Detected mode: real (d_model=1024, heads=16)
[Tension] Starting tension sweep
[Tension] Target layers: [0, 4, 12, 24]
[Tension] Epsilon low: 0.001, Epsilon high: 0.1
[Tension] Alpha values: [0.0, 0.25, 0.5, 0.75, 1.0]

Tension Sweep: 100%|██████████| 24/24 [06:21<00:00, 15.90s/it]

[Tension] Computing baseline metrics...
 Layer 0: eff_rank=624.0
 Layer 4: eff_rank=603.9
 Layer 12: eff_rank=582.7
 Layer 24: eff_rank=694.7

[Tension] Running single-head baseline...
[Tension] Running dual-head experiments...
[Tension] Results saved to results\tension_sweep.json
[Tension] Total results: 24
[Tension] Comparison summary:
 Layer 0: best_alpha=None
 Layer 4: best_alpha=None
 Layer 12: best_alpha=None
 Layer 24: best_alpha=None

[Step 2b] Done! 24 results collected
```

---

## Resultados por Alpha

### Single-Head Baseline

| Layer | ε=0.001 | ε=0.01 | ε=0.1 |
|-------|---------|--------|-------|
| 0 | 606.7 | 580.7 | 355.0 |
| 4 | 606.7 | 580.7 | 355.0 |
| 12 | 606.7 | 580.7 | 355.0 |
| 24 | 606.7 | 580.7 | 355.0 |

### Dual-Head Results

#### α = 0.0 (Pure High-ε Head)

| Layer | Effective Rank | Δ vs Baseline |
|-------|----------------|----------------|
| 0 | 355.0 | -43.1% |
| 4 | 355.0 | -41.2% |
| 12 | 355.0 | -39.1% |
| 24 | 355.0 | -48.9% |

**Interpretación**: La cabeza de alta expressividad (ε=0.1) produce attention casi uniforme, perdiendo mucha dimensionalidad.

#### α = 0.25 (25% Low-ε)

| Layer | Effective Rank | Δ vs Baseline |
|-------|----------------|----------------|
| 0 | 605.1 | -3.0% |
| 4 | 605.1 | +0.2% |
| 12 | 605.1 | +3.9% |
| 24 | 605.1 | -12.8% |

**Interpretación**: Un poco de concentración mejora la dimensionalidad en capas medias.

#### α = 0.50 (Balance)

| Layer | Effective Rank | Δ vs Baseline |
|-------|----------------|----------------|
| 0 | 606.2 | -2.9% |
| 4 | 606.2 | +0.4% |
| 12 | 606.2 | +4.0% |
| 24 | 606.2 | -12.6% |

**Interpretación**: Balance equilibrado, similar a single-head ε=0.001.

#### α = 0.75 (75% Low-ε)

| Layer | Effective Rank | Δ vs Baseline |
|-------|----------------|----------------|
| 0 | 606.6 | -2.8% |
| 4 | 606.6 | +0.4% |
| 12 | 606.6 | +4.1% |
| 24 | 606.6 | -12.6% |

**Interpretación**: Casi igual a α=0.5, la cabeza de concentración domina.

#### α = 1.0 (Pure Low-ε Head)

| Layer | Effective Rank | Δ vs Baseline |
|-------|----------------|----------------|
| 0 | 606.7 | -2.7% |
| 4 | 606.7 | +0.5% |
| 12 | 606.7 | +4.1% |
| 24 | 606.7 | -12.7% |

**Interpretación**: Idéntico al single-head ε=0.001.

---

## Análisis

### Observación Clave

**El dual-head no mejora sobre el single-head óptimo**:

```
Single-head ε=0.001:  eff_rank = 606.7
Dual-head α=0.0:      eff_rank = 355.0  (peor)
Dual-head α=0.25:     eff_rank = 605.1  (similar)
Dual-head α=0.5:      eff_rank = 606.2  (similar)
Dual-head α=0.75:     eff_rank = 606.6  (similar)
Dual-head α=1.0:      eff_rank = 606.7  (igual)
```

### Conclusión

Para este experimento:

1. **No hay beneficio en dual-head**: El single-head con ε=0.001 ya es óptimo
2. **La cabeza high-ε es contraproducente**: ε=0.1 produce attention demasiado uniforme
3. **El α no afecta significativamente**: Entre 0.25 y 1.0, los resultados son prácticamente idénticos

---

## ¿Por qué no funciona el Dual-Head?

### Hipótesis

1. **Gap demasiado grande entre epsilons**:
   - ε_low = 0.001 produce attention peaked
   - ε_high = 0.1 produce attention semi-uniforme
   - La combinación no añade valor

2. **La concentración óptima es single-head**:
   - ε=0.001 logra el mejor trade-off solo
   - No hay necesidad de una segunda cabeza

3. **Las capas de Qwen3-0.6B ya son comprimidas**:
   - El modelo 4-bit ya tiene representaciones compactas
   - Más concentración no ayuda

---

## Visualización Generada

```
plots/tension_alpha_vs_rank.png
```

Muestra la relación entre α y effective rank para cada capa.

---

## JSON de Resultados

```json
{
  "experiment": "Tension Sweep (Dual-Head)",
  "mode": "real",
  "config": {
    "epsilon_low": 0.001,
    "epsilon_high": 0.1,
    "alpha_values": [0.0, 0.25, 0.5, 0.75, 1.0],
    "target_layers": [0, 4, 12, 24],
    "d_model": 1024,
    "num_heads": 16
  },
  "baseline_ranks": {
    "0": 624.0,
    "4": 603.9,
    "12": 582.7,
    "24": 694.7
  },
  "best_alpha": {
    "0": null,
    "4": null,
    "12": null,
    "24": null
  }
}
```

---

## Recomendación

**Usar single-head con ε=0.001**. El dual-head no aporta beneficios en este contexto.

El dual-head podría ser útil cuando:
- Se quiere explorar trade-offs más finos
- El sweet spot no es claro
- Se requiere control granular de concentración vs expressividad

---

## Próximo Paso

→ Ver [05-metricas-finales.md](./05-metricas-finales.md) para el análisis completo de métricas.
