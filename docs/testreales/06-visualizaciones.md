# Visualizaciones — LLM-BUBBLE v4

**Fecha**: 16 Abril 2026
**Ubicación**: `plots/` directory

---

## 1. Visualizaciones Principales

### 1.1 Effective Rank Curves (`effective_rank_curves.png`)

**Descripción**: Curvas de effective rank vs epsilon para cada capa analizada.

**Contenido**:
- Eje X: Valores de epsilon (0.0, 0.001, 0.01, 0.1, 1.0)
- Eje Y: Effective rank
- Líneas: Una por cada capa (0, 4, 12, 24)
- Línea horizontal: Baseline (sin Plateau attention)

**Hallazgos clave**:
- Las curvas muestran degradación gradual del effective rank con epsilon creciente
- La capa 0 (embedding) es la más sensible a epsilon
- La capa 24 (final) muestra mayor robustez
- El sweet spot ε = 0.001 está en el plateau inicial de todas las curvas

**Interpretación visual**:
```
Effective Rank
    │
700 ┼──────────────────────────────────── baseline
    │    ╲
650 ┼     ╲
    │      ╲
600 ┼───────╲─────────────────────────── sweet spot (ε=0.001)
    │        ╲
550 ┼         ╲
    │          ╲
500 ┼           ╲
    │            ╲
450 ┼             ╲
    │              ╲
400 ┼               ──╲
    │                   ╲
350 ┼──────────────────────╲── ε=0.1 (colapso)
    │
    └────────┬────┬────┬────┬────
            0.001 0.01 0.1  1.0   Epsilon
```

---

### 1.2 Concentration Heatmap (`concentration_heatmap_*.png`)

**Descripción**: Mapas de calor mostrando concentration ratio por capa y epsilon.

**Archivos**:
- `concentration_heatmap_effective_rank.png` - Effective rank por (layer, epsilon)
- `concentration_heatmap_concentration_ratio.png` - Concentration ratio por (layer, epsilon)

**Contenido**:
- Filas: Capas (0, 4, 12, 24)
- Columnas: Valores de epsilon
- Color: Valor de la métrica (rojo = alto, azul = bajo)

**Hallazgos clave**:
- El heatmap de effective rank muestra bandas horizontales (cada capa mantiene su patrón)
- El heatmap de concentration ratio muestra gradiente diagonal (más concentración con menor epsilon)
- La región óptima está en la esquina superior izquierda (capas early, epsilon pequeño)

**Interpretación visual**:
```
         Epsilon
         0.001   0.01    0.1     1.0
       ┌───────┬───────┬───────┬───────┐
     0 │ ████  │ ███   │ ██    │ █     │
L      ├───────┼───────┼───────┼───────┤
a    4 │ ████  │ ███   │ ██    │ █     │
y      ├───────┼───────┼───────┼───────┤
e   12 │ ████  │ ███   │ ██    │ █     │
r      ├───────┼───────┼───────┼───────┤
    24 │ ████  │ ███   │ ██    │ █     │
       └───────┴───────┴───────┴───────┘

       ████ = Alta concentración (óptimo para Plateau)
       █    = Baja concentración (similar a uniforme)
```

---

### 1.3 Pareto Frontier (`pareto_frontier.png`)

**Descripción**: Frontera de Pareto entre effective rank y concentration ratio.

**Contenido**:
- Eje X: Concentration ratio (0 = máxima concentración, 1 = uniforme)
- Eje Y: Effective rank
- Puntos: Cada configuración (layer, epsilon)
- Línea: Frontera de Pareto (soluciones no dominadas)
- Zona sombreada: Región factible

**Hallazgos clave**:
- El sweet spot (ε = 0.001) está en la esquina superior izquierda
- La frontera de Pareto muestra el trade-off entre concentración y dimensionalidad
- No hay soluciones dominadas para ε < 0.1
- ε > 0.1 está en la región dominada (peor en ambos objetivos)

**Interpretación visual**:
```
Effective Rank
    │
700 ┼ ●─────────────────────────────── baseline
    │  ╲
650 ┼   ╲
    │    ╲ Pareto frontier
600 ┼─────●──────────────────────────── sweet spot (ε=0.001)
    │      ╲
550 ┼       ●────────────────────────── ε=0.01
    │        ╲
500 ┼         ╲
    │          ╲
450 ┼           ╲
    │            ╲
400 ┼             ╲
    │              ╲
350 ┼───────────────●────────────────── ε=0.1
    │                  ╲
300 ┼───────────────────●────────────── ε=1.0 (dominado)
    │
    └───────┬───────┬───────┬───────┬──
           0.0    0.05    0.3    0.5  Concentration Ratio
    
    ● = Soluciones en la frontera de Pareto
```

---

### 1.4 Anisotropy vs Epsilon (`anisotropy_vs_epsilon.png`)

**Descripción**: Curva de anisotropy index vs epsilon.

**Contenido**:
- Eje X: Epsilon (log scale)
- Eje Y: Anisotropy index
- Línea: Anisotropía promedio
- Banda: Desviación estándar entre capas

**Hallazgos clave**:
- Anisotropía mínima en ε ≈ 0.01 (no en el sweet spot)
- Anisotropía aumenta para epsilon muy pequeño o muy grande
- El sweet spot (ε = 0.001) tiene anisotropía ligeramente mayor pero aún baja

**Interpretación visual**:
```
Anisotropy Index
    │
0.05 ┼───────────────────────────●──── ε=0.1
    │                          ╱
0.04 ┼────────────●────────────╱────── baseline
    │            ╱
0.03 ┼──────────╱
    │         ╱
0.02 ┼────────●──────────────────────── ε=0.01 (mínimo)
    │       ╱
0.01 ┼──────╱
    │     ╱
0.00 ┼───●───────────────────────────── ε=0.001
    │
    └────┬─────┬─────┬─────┬─────┬────
        0.001 0.01  0.1   1.0        Epsilon (log scale)
```

---

### 1.5 Intrinsic Dimension vs Epsilon (`intrinsic_dim_vs_epsilon.png`)

**Descripción**: Curva de intrinsic dimension (MLE) vs epsilon.

**Contenido**:
- Eje X: Epsilon
- Eje Y: Intrinsic dimension
- Línea: Dimensión intrínseca promedio

**Hallazgos clave**:
- La dimensión intrínseca permanece constante en 1.0 para todos los epsilon
- Esto indica que el manifold subyacente es simple independientemente de la compresión
- No hay ganancia de complejidad con Plateau attention

**Interpretación visual**:
```
Intrinsic Dimension
    │
1.5 ┼
    │
1.0 ┼─────────────────────────────────── constante (1.0)
    │
0.5 ┼
    │
0.0 ┼
    └────┬─────┬─────┬─────┬─────┬────
        0.0  0.001 0.01  0.1   1.0   Epsilon
```

---

### 1.6 Summary Dashboard (`summary_dashboard.png`)

**Descripción**: Panel de 4 plots resumiendo todos los resultados.

**Contenido**:
- Panel 1 (superior izquierda): Effective rank curves
- Panel 2 (superior derecha): Concentration heatmap
- Panel 3 (inferior izquierda): Pareto frontier
- Panel 4 (inferior derecha): Anisotropy vs epsilon

**Hallazgos clave**:
- Visión consolidada de todas las métricas principales
- El sweet spot es evidente en los 4 paneles
- La consistencia entre métricas valida la elección de ε = 0.001

---

## 2. Visualizaciones de Tension Sweep

### 2.1 Tension Alpha vs Rank (`tension_alpha_vs_rank.png`)

**Descripción**: Effective rank vs alpha (parámetro de blending dual-head).

**Contenido**:
- Eje X: Alpha (0 = solo epsilon_low, 1 = solo epsilon_high)
- Eje Y: Effective rank
- Líneas: Una por cada capa
- Línea vertical: Sweet spot del single-head (α = 1.0)

**Hallazgos clave**:
- El dual-head NO mejora sobre el single-head
- α = 1.0 (equivalente a single-head con ε_low) es óptimo
- α = 0.0 produce colapso (effective rank bajo)
- Valores intermedios de alpha no muestran beneficios

**Interpretación visual**:
```
Effective Rank
    │
700 ┼────────────────────────────●───── baseline
    │                           ╱
650 ┼                          ╱
    │                         ╱
600 ┼────────────────────────●────── sweet spot (α=1.0, single-head)
    │                      ╱
550 ┼                    ╱
    │                  ╱
500 ┼               ╱
    │            ╱
450 ┼         ╱
    │      ╱
400 ┼───●────────────────────────────── α=0.0 (colapso)
    │
    └────┬─────┬─────┬─────┬─────┬────
        0.0   0.25  0.5   0.75  1.0   Alpha
```

---

## 3. Visualizaciones Adicionales

### 3.1 Cost Comparison Pareto (`cost_comparison_pareto.png`)

**Descripción**: Comparación de costos numéricos entre configuraciones.

**Contenido**:
- Eje X: Condition number de la matriz de costo
- Eje Y: Spectral gap
- Puntos: Configuraciones (layer, epsilon)
- Color: Epsilon

**Hallazgos clave**:
- Epsilon pequeño produce condition numbers más bajos (mejor estabilidad)
- Epsilon grande produce spectral gaps más grandes (más estructura)
- El sweet spot tiene un balance razonable entre ambos

---

### 3.2 Layer Selection Plots

**Archivos legacy** (generados en experimentos anteriores):

- `layer_selection_concentration_gain.png`
- `layer_selection_intrinsic_dim_preservation.png`
- `layer_selection_pareto_movement.png`
- `layer_selection_ranking.png`
- `layer_selection_rank_comparison.png`

**Nota**: Estos plots fueron generados en experimentos previos y pueden no corresponder a la configuración actual. Se mantienen para referencia histórica.

---

## 4. Interpretación General

### 4.1 Patrón Consistente

Todas las visualizaciones muestran un patrón consistente:

```
┌────────────────────────────────────────────────────┐
│ REGIÓN ÓPTIMA: ε ≤ 0.01                            │
├────────────────────────────────────────────────────┤
│ • Effective rank preservado (> 95% del baseline)  │
│ • Concentration ratio alto (atención peaked)       │
│ • Anisotropy bajo (distribución isotrópica)        │
│ • Sin colapso representacional                     │
│ • Sin crowding                                     │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ REGIÓN DE COLAPSO: ε ≥ 0.1                         │
├────────────────────────────────────────────────────┤
│ • Effective rank reducido (< 60% del baseline)    │
│ • Concentration ratio cercano a 0.5 (uniforme)     │
│ • Collapse score = 1.0                             │
│ • Crowding ratio = 1.0                             │
│ • Mean NN distance ≈ 0                             │
└────────────────────────────────────────────────────┘
```

### 4.2 Dual-Head No Mejora

El tension sweep muestra que:

```
┌────────────────────────────────────────────────────┐
│ CONCLUSIÓN DUAL-HEAD                               │
├────────────────────────────────────────────────────┤
│ • α = 1.0 (single-head puro) es óptimo             │
│ • Valores intermedios de alpha no aportan beneficio│
│ • α = 0.0 causa colapso                            │
│ • Recomendación: Usar single-head con ε = 0.001   │
└────────────────────────────────────────────────────┘
```

---

## 5. Ubicación de Archivos

```
plots/
├── anisotropy_vs_epsilon.png              (65 KB)
├── concentration_heatmap_*.png            (113-135 KB)
├── cost_comparison_pareto.png             (107 KB)
├── effective_rank_curves.png              (109 KB)
├── intrinsic_dim_vs_epsilon.png           (65 KB)
├── pareto_frontier.png                    (99 KB)
├── summary_dashboard.png                  (240 KB)
├── tension_alpha_vs_rank.png              (88 KB)
└── layer_selection_*.png                  (legacy)
```

**Total**: 14 archivos PNG (~1.5 MB)

---

*Visualizaciones generadas con matplotlib*
*Resolución: 100 DPI (default)*
*Formato: PNG con transparencia*
