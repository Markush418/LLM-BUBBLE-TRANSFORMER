# BT V5 · Paper IV — Complejidad y Scaling
### Derivación del O(N·C), speedup honesto, y comparación con el estado del arte

> **Qué es esto.** El análisis cuantitativo: de dónde sale el O(N·C), la fórmula de speedup y su corrección por Amdahl (el número honesto vs el número de marketing), la comparación cabeza a cabeza con LOTFormer, la adopción de FlashSinkhorn, y el comportamiento al escalar a modelos grandes.
>
> **Regla de oro del documento:** todo número de speedup debe presentarse con su contexto. El 124× es teórico (FLOPs); el ~54% es el impacto en el modelo completo (Amdahl); el 2.02× es el único wall-clock real medido. Confundirlos es deshonesto y un reviewer lo penaliza.
>
> **Tags:** `[THM]` · `[EMPIRICAL]` · `[HEURISTIC]` · `⚠`.

---

## §1 — Complejidad del forward pass

[THM] **Costo por bloque** (notación: $N$ = longitud de secuencia, $C$ = nº burbujas, $d$ = dimensión, $|b|$ = tokens en burbuja $b$):

| Bloque | Costo | Nota |
|---|---|---|
| Proyecciones Q,K,V | $O(N d^2)$ | igual que cualquier Transformer |
| RoPE | $O(Nd)$ | rotación elemento a elemento |
| Routing token→burbuja | $O(NC)$ | $N$ tokens × $C$ centroides |
| Atención intra-burbuja | $O\big(\sum_b|b|^2\big)$ | densa dentro de cada celda |
| Agregación | $O(Nd^2)$ | proyección de salida |

[THM] **Dos regímenes según el supuesto de balance:**

- **Balanceado** ($|b|\approx N/C$): la atención intra-burbuja cuesta $\sum_b (N/C)^2 = C\cdot N^2/C^2 = N^2/C$. Total dominante: $O(NC + N^2/C)$.
  - Minimizando en $C$: $\frac{d}{dC}(NC+N^2/C)=0 \Rightarrow C^\star=\sqrt N \Rightarrow$ **$O(N^{1.5})$**.
  - Para $C$ fijo y pequeño: el término $N^2/C$ domina pero con constante $1/C$ → speedup $\sim C$ vs baseline.
- **Centroide-a-token** (variante O(N·C) pura): si en lugar de atención densa intra-burbuja se hace atención de cada token contra los $C$ **centroides** (no contra los miembros), el costo es $O(NC)$ **lineal en N**. Esta es la variante que compite con LOTFormer.

⚠ La distinción entre "atención intra-burbuja densa" (O(N^{1.5})) y "atención token-a-centroide" (O(NC)) es **decisiva** y debe declararse explícitamente. El BT puede operar en cualquiera de los dos modos; sus garantías de complejidad difieren.

---

## §2 — Fórmula de speedup y su corrección honesta

[THM] **Speedup de la atención** (FLOPs, vs baseline O(2N²)):
$$\operatorname{speedup}(N,C)=\frac{2N^2}{N\!\cdot\! C + N^2/C}.$$
- $C^\star=\sqrt N \Rightarrow \operatorname{speedup}_{\max}=\sqrt N$.
- Para $C$ fijo: $\operatorname{speedup}\to N/C$ (crece linealmente con $N$).

[HEURISTIC] **Corrección de Amdahl (impacto en el modelo completo).** La atención no es el 100% del cómputo; el resto (MLP, embeddings, LM head) no se acelera. Con fracción de atención $\approx 0.35$:
$$\operatorname{speedup}_{\text{modelo}}\approx\frac{1}{0.65 + 0.35/\operatorname{speedup}_{\text{aten}}}.$$

[EMPIRICAL] **Tabla honesta** ($C=64$):

| $N$ | speedup atención (FLOPs) | speedup modelo completo (Amdahl) |
|---|---|---|
| 8K | ~56× | ~1.48× |
| 32K | ~114× | ~1.53× |
| 128K | ~124× | ~1.54× |

⚠ **Regla de presentación:** el 124× **debe** aparecer junto al ~1.54× modelo-completo, nunca solo. En el reporte HTML actual la aclaración está enterrada → cualquier reviewer/periodista toma el 124× fuera de contexto. Presentar siempre el par **(124× atención, ~54% mejora modelo completo @ 128K)**.

[EMPIRICAL] **El único número wall-clock real:**
```
Config: d=512, heads=8, C=32, CPU, B=2, N=128
V2-Sinkhorn: 36.77 ms    V3-SDOT: 18.23 ms    →  2.02× medido
```
Todo lo demás es teórico (FLOPs, no wall-clock). **Explicitar esta distinción** es obligatorio: 2.02× = real medido; 124× = analítico.

⚠ **Supuesto no medido:** la fórmula asume distribución uniforme de tokens por burbuja. En texto natural la distribución Voronoi es sesgada (tokens frecuentes → burbujas grandes) → degrada speedup empírico vs teórico. **Acción Sprint 3:** medir distribución real y reportar varianza del speedup.

---

## §3 — Comparación cabeza a cabeza con el estado del arte

[THM] **Tabla de complejidad** (del survey, contextualizada):

| Método | Complejidad | Mecanismo | Relación con BT |
|---|---|---|---|
| Atención estándar | $O(N^2)$ | softmax all-pairs | baseline a vencer |
| Sinkformer | $>O(N^2)$ | Sinkhorn completo | ancestro OT; el Impuesto de Jersey lo hunde |
| **LOTFormer** | $O(Nr)$ | OT low-rank vía pivote | **competidor directo** |
| Routing Transformer | $O(N^{1.5})$ | k-means online | análogo clásico de burbujas |
| NSA / MoBA | $O(N\!\cdot\!k)$ | sparse/block trainable | rivales de producción |
| Mamba-3 / Gated DeltaNet | $O(N)$ | recurrencia lineal/SSM | paradigma dominante |
| **BT V5** | $O(NC)$ ó $O(N^{1.5})$ | SDOT sobre Power Diagram | novedad = geometría + SIRI |

[ARCH] **BT vs LOTFormer — la diferenciación crítica:**

| Dimensión | BT V5 (SDOT) | LOTFormer |
|---|---|---|
| Fundamento | $\varepsilon\to 0$ = Voronoi (límite **exacto** de softmax) | factorización low-rank OT vía pivote |
| Complejidad | $O(NC)$ (un paso geométrico) | $O(Nr)$ (dos problemas OT por capa) |
| Estocasticidad | one-sided (fila) | doubly stochastic |
| Iteraciones | **ninguna** (Voronoi argmin) | composición de dos transportes |
| Claim | límite matemático de softmax | aproximación con garantías |

[ARCH] **El argumento de venta del BT frente a LOTFormer:** LOTFormer logra tiempo lineal pero **requiere dos problemas OT por capa** (query→pivote, pivote→key). El BT logra $O(NC)$ con **un solo paso geométrico** (Voronoi argmin) — sin iteraciones, sin composición de transportes. La pregunta empírica que decide todo: ¿el BT vence a LOTFormer en LRA accuracy *y* en velocidad wall-clock? (Paper V).

⚠ **Si el BT no vence a LOTFormer en ambos ejes, la recomendación del survey es construir sobre la formulación de pivotes de LOTFormer en vez de mantener el solver OT a medida.** Honestidad estratégica: LOTFormer ya publicó lo que el BT intenta; la única defensa es ser estrictamente mejor o estructuralmente más simple con métricas que lo respalden.

---

## §4 — FlashSinkhorn: el kernel IO-aware

[FIX] *(arXiv:2602.03067)* FlashSinkhorn reescribe las actualizaciones de Sinkhorn estabilizadas en log-domain como reducciones **LogSumExp por filas** de scores dot-product sesgados — exactamente la normalización de la atención → habilita tiling/fusión estilo FlashAttention. Reporta hasta **32× forward** y **161× end-to-end** sobre baselines online en A100.

[ARCH] **Relevancia para el BT:** aunque el BT V5 evita las iteraciones de Sinkhorn en el forward (usa Power Diagram directo), las usa en **entrenamiento** (Gumbel-Sinkhorn, Paper III §1). FlashSinkhorn hace ese paso IO-eficiente. ⚠ Limitación: por ahora solo costo euclidiano al cuadrado — compatible con $C_{ij}=\|Q_i-K_j\|^2$ del BT, pero **no** con costos geodésicos (Poincaré). Para los manifolds curvos, el kernel no aplica directamente.

---

## §5 — Scaling a modelos grandes (7B–405B)

[HEURISTIC] **Overhead de centroides.** El BT añade $C\times d$ parámetros de centroides + $C$ pesos $\psi$ por capa. Para $C=64$, $d=4096$: ~262K params/capa → despreciable frente a los ~50M params/capa de un 7B. El overhead es sub-1%.

[HEURISTIC] **Compatibilidad con arquitecturas de producción:**
- **GQA** (Grouped-Query Attention): el BT aplica RoPE pre-expansion GQA (Paper III §3.3) → compatible. El clustering opera sobre los KV-heads agrupados → **el overhead de centroides es ~8× menor** porque hay menos KV-heads que query-heads.
- **MoE:** ortogonal — el BT reemplaza la atención, MoE reemplaza el MLP. Componen.
- **MLA** (DeepSeek): ⚠ **conflicto** — MLA comprime el KV a un latente de bajo rango → distorsiona la geometría del manifold → invalida las métricas geodésicas del BT. Por eso DeepSeek está excluido de la pila de prueba. En modo euclidiano (sin geoopt) podrían coexistir, pero pierde el sustento geométrico.
- **Thinking mode** (Qwen3-32B): secuencias de razonamiento muy largas → es justo donde el $O(NC)$ del BT más gana frente al $O(N^2)$.

[ARCH] **Lección de scaling:** el BT escala bien en parámetros (overhead sub-1%) y mejor cuanto más largo el contexto (el speedup crece con $N$). El riesgo no es el scaling — es la **competencia** (LOTFormer, híbridos lineales) que ya opera a escala de producción.

---

## §6 — Resumen cuantitativo

```
COMPLEJIDAD:
  • token-a-centroide:  O(N·C)        ← compite con LOTFormer O(N·r)
  • intra-burbuja densa: O(N^1.5)     ← C* = √N
  • baseline:            O(N²)

SPEEDUP (honesto, C=64, N=128K):
  • atención (FLOPs):       124×
  • modelo completo (Amdahl): ~1.54× (≈54% mejora)
  • wall-clock real medido:  2.02× (V2→V3, N=128, CPU)

PARÁMETROS EXTRA: sub-1% (C·d + C por capa)
SCALING: mejora con N; overhead sub-1%; ⚠ incompatible con MLA en modo geodésico
```

[ARCH] **El número que importa para el paper:** no el 124×, sino el **ΔPPL ≤ 2%** (Paper V). Un speedup espectacular con regresión de PPL es inútil. El orden de prioridad es: (1) preservar calidad predictiva, (2) demostrar speedup wall-clock real, (3) vencer a LOTFormer. En ese orden.

---

*Siguiente: Paper V — Protocolo experimental, posicionamiento competitivo, y checklist arXiv.*
