# BT V5 · Paper I — Arquitectura
### El forward pass real del nuevo Bubble Transformer

> **Qué es esto.** La especificación formal de la arquitectura BT V5: la síntesis de V4 (SDOT + Power Diagram + Expert-Choice + geoopt) con todos los fixes y mejoras derivados de este chat — Gumbel-Sinkhorn para diferenciabilidad, reconciliación MoSA del routing, RoPE integrado en orden correcto, capas híbridas, y FlashSinkhorn como kernel. Cada bloque incluye su definición, su complejidad y su justificación.
>
> **Relación con versiones previas.** V3 = SDOT funcional (Voronoi duro, O(N log C), 2.02× verificado). V4 = + Expert-Choice + Laguerre + geoopt + FPS (con 2 blockers abiertos). **V5 = V4 con los blockers cerrados + las mejoras del survey 2025-2026.**
>
> **Tags:** `[DEF]` · `[THM]` · `[ARCH]` decisión arquitectónica · `[FIX]` cierra un blocker · `[NEW]` aporte del chat · `⚠` riesgo.

---

## §1 — Visión general del forward pass

El BT V5 reemplaza el bloque `self_attn` de un Transformer host (Qwen3-0.6B en debug) por el módulo `BubbleAttentionV5`. El pipeline, de entrada a salida:

```
hidden_states  x ∈ ℝ^{B×N×d}
   │
   ├─[1] Proyecciones Q,K,V                         O(N·d²)
   ├─[2] RoPE sobre Q,K (post-proyección)   [FIX]   O(N·d)
   ├─[3] Selección de C (Baroreceptor MLP)  [NEW]   O(N·d)
   ├─[4] Centroides {c_i}: FPS init + geoopt        O(N·C) init
   ├─[5] Power Diagram (Laguerre, pesos ψ)          O(N·C)
   ├─[6] Asignación token→burbuja diferenciable
   │     vía Gumbel-Sinkhorn                 [FIX]   O(N·C·τ_GS)
   ├─[7] SDOT intra-burbuja (one-sided EOT)         O(N·C + Σ|b|²)
   ├─[8] Routing Expert-Choice reconciliado
   │     (MoSA: la burbuja elige top-k)     [FIX]   O(N·C)
   └─[9] Agregación → output ∈ ℝ^{B×N×d}            O(N·d²)
```

Complejidad dominante: **O(N·C)** con $C\ll N$ (frente a O(N²) del baseline). Detalle completo en Paper IV.

---

## §2 — Proyecciones y RoPE [FIX del incidente 831,974]

[DEF] Proyecciones lineales estándar: $Q=xW_Q$, $K=xW_K$, $V=xW_V$, con reshape a heads $[B,H,N,d_h]$.

[FIX] **RoPE se aplica a $Q,K$ después de la proyección y antes del clustering** (§5–6). El orden es crítico y fue la causa raíz del incidente PPL = 831,974:
$$Q\leftarrow R_\Theta(\text{pos})\,Q,\qquad K\leftarrow R_\Theta(\text{pos})\,K,$$
donde $R_\Theta(\text{pos})$ es la rotación de RoPE dependiente de posición (ver Documento 0, Capa A; detalle en Paper III §3).

[ARCH] **Por qué el orden importa.** El routing geométrico (Voronoi/Laguerre) opera en el espacio $Q$-$K$. Si $Q,K$ no llevan información posicional, los centroides aprenden clusters **solo semánticos** → las burbujas son posicionalmente incoherentes → los logits colapsan. La consecuencia formal: las métricas geométricas puras ($R_{\text{eff}}$, concentration ratio) siguen válidas sin RoPE, pero la **PPL** no, porque la posición es predictivamente esencial.

⚠ Constraints heredados de Qwen3 (transformers 4.51.0): firma `rotary_emb(x, position_ids)`; aplicar RoPE **antes** de la expansión GQA; retornar exactamente `(output, None)` del wrapper.

---

## §3 — Baroreceptor MLP: selección dinámica de C [NEW]

[NEW] En V3/V4 el número de burbujas $C$ es un hiperparámetro fijo. En V5 se introduce el **Baroreceptor MLP**: una red pequeña que predice $C$ por capa/secuencia a partir de un estadístico de la distribución de atención.

[DEF] Sea $\hat H$ la entropía empírica de la distribución de afinidades de la capa. El Baroreceptor mapea
$$C = \operatorname{round}\big(C_{\min} + (C_{\max}-C_{\min})\cdot\sigma(\text{MLP}(\hat H,\ \text{stats}))\big).$$

[ARCH] **Fundamento teórico (phase transitions).** La literatura de transiciones de fase en atención (arXiv:2601.19942, arXiv:2510.07401) identifica dos regímenes separados por una profundidad crítica $\gamma_c\approx 0.42$:
- **Régimen líquido** (alta entropía, bulk Marchenko–Pastur): atención difusa → conviene $C$ alto (más burbujas pequeñas).
- **Régimen sólido** (baja entropía, spectral gaps): atención concentrada en cuencas → conviene $C$ bajo.

El Baroreceptor es, formalmente, **un detector de régimen líquido/sólido** que ajusta $C$ antes de la transición. Esto convierte la "conjetura abierta" de V4 en un mecanismo implementable.

⚠ Status: `[CONJECTURE]` con respaldo de tres papers independientes. Requiere medir $\beta_0,\beta_1$ del grafo de atención $G_\varepsilon$ durante el training para validar (Sprint 3-4, ver Paper V).

---

## §4 — Centroides: FPS init + optimización Riemanniana

[DEF] **Inicialización por Farthest Point Sampling (FPS).** Se eligen $C$ centroides iniciales maximizando la cobertura geométrica: $c_1$ aleatorio, $c_{i+1}=\arg\max_x\min_{j\le i}d(x,c_j)$. Garantiza dispersión inicial (evita centroides colapsados).

[DEF] **Centroides como parámetros Riemannianos.** $c_i\in\mathcal M$ (manifold), declarados como `geoopt.ManifoldParameter`, optimizados con `RiemannianAdam`. V5 soporta 4 manifolds: euclidiano $\mathbb{R}^d$, esfera $\mathbb{S}^{d-1}$, Stiefel, y **Disco de Poincaré** (hiperbólico, para jerarquías semánticas).

[ARCH] La elección de manifold depende del host: RoPE induce naturalmente estructura de Stiefel (Llama-3.2-1B); jerarquías semánticas favorecen Poincaré (TinyLlama). ⚠ DeepSeek queda **excluido**: su MLA comprime el KV → distorsiona el manifold → invalida las métricas geodésicas.

---

## §5 — Power Diagram (celdas de Laguerre) con pesos ψ

[DEF] Cada burbuja $i$ tiene centroide $c_i$ y **peso aprendible** $\psi_i\in\mathbb{R}$. La asignación se hace por **power distance** (Documento 0, Capa A):
$$\operatorname{pow}(x,c_i)=\|x-c_i\|^2-\psi_i,\qquad b(x)=\arg\min_i\operatorname{pow}(x,c_i).$$

[THM] *(Justificación, Documento 0 Capa F)* En el límite $\varepsilon\to 0$ del OT entrópico one-sided, la matriz de atención converge **exactamente** a la teselación de Laguerre inducida por $\{(c_i,\psi_i)\}$. El BT no aproxima softmax: **es** su límite $\varepsilon\to 0$ computado geométricamente en un paso, sin las iteraciones de Sinkhorn (sin Impuesto de Jersey).

[ARCH] Los pesos $\psi_i$ dan **burbujas de tamaño variable** adaptadas a la densidad semántica: regiones densas → muchas burbujas pequeñas; regiones ralas → pocas grandes. Esto generaliza el $k$-means (Voronoi no ponderado) del Routing Transformer.

---

## §6 — Asignación diferenciable vía Gumbel-Sinkhorn [FIX Blocker 1]

⚠ **El blocker.** $b(x)=\arg\min_i\operatorname{pow}(x,c_i)$ es escalonado → gradiente cero/indefinido → los pesos $\psi_i$ y centroides $c_i$ no entrenan por backprop.

[FIX] **Gumbel-Sinkhorn** (Mena et al. 2018, arXiv:1802.08665). En lugar del argmin duro, se construye una matriz de asignación soft doblemente estocástica:
$$\Pi=\operatorname{Sinkhorn}_\tau\big(-\operatorname{pow}(x,c)+g\big),\qquad g\sim\operatorname{Gumbel}(0,1),$$
con temperatura $\tau$. $\Pi\to$ asignación dura (permutación de bloque) cuando $\tau\to 0$; es diferenciable para $\tau>0$.

[ARCH] **Por qué Gumbel-Sinkhorn y no Gumbel-Softmax plano.** La asignación token→burbuja tiene estructura de **acoplamiento** (restricción de capacidad por burbuja), no categórica independiente. Gumbel-Sinkhorn la respeta usando el **mismo aparato de Sinkhorn** que el BT ya tiene → cero maquinaria nueva. Alternativa: **MESH** (Minimize Entropy of Sinkhorn), que infla el costo para que la solución entrópica sea casi one-hot manteniéndose diferenciable. Detalle de gradientes en Paper III §1.

[ARCH] **Straight-Through.** Forward usa la asignación dura (preserva la complejidad O(N·C) y la sparsity de bloque); backward usa el gradiente del $\Pi$ soft. Lo mejor de ambos: cómputo duro, aprendizaje suave.

---

## §7 — SDOT intra-burbuja (atención one-sided EOT)

[DEF] Dentro de cada burbuja $b$ con tokens $\{x_j\}_{j\in b}$, se computa atención densa estándar (one-sided EOT, i.e. softmax) **solo entre los miembros de la burbuja**:
$$\operatorname{Attn}_b(Q_b,K_b,V_b)=\operatorname{softmax}\!\Big(\tfrac{Q_bK_b^\top}{\sqrt{d}}\Big)V_b.$$
Cero atención inter-burbuja. La matriz global es **sparse por bloque**.

[THM] Complejidad: $\sum_b|b|^2$. Con burbujas balanceadas $|b|\approx N/C$, esto es $C\cdot(N/C)^2=N^2/C$. Sumado al costo de routing $O(N\cdot C)$, el total es $O(N\cdot C + N^2/C)$, minimizado en $C=\sqrt N$ → **$O(N^{1.5})$** (igual que Routing Transformer en el peor caso balanceado, mejor con $C$ chico). Análisis completo y la versión O(N·C) con atención centroide-a-token en Paper IV.

⚠ **Supuesto de balance.** La fórmula asume distribución uniforme de tokens entre burbujas. En texto natural la distribución Voronoi es sesgada (tokens frecuentes → burbujas grandes) → degrada el speedup empírico. Medir la varianza real es tarea de Sprint 3 (Paper V).

---

## §8 — Routing Expert-Choice reconciliado [FIX Blocker 2]

⚠ **El blocker.** Expert-Choice (Zhou et al. 2022, arXiv:2202.09368) = "los expertos eligen top-k tokens" → balance perfecto sin auxiliary loss. Pero una partición Power-Diagram **dura** asigna cada token a **una** celda (token-choice de capacidad 1) → estructuralmente incompatible con Expert-Choice.

[FIX] **Adoptar la formulación MoSA** (Mixture of Sparse Attention, arXiv:2505.00315): cada burbuja/experto $i$ selecciona **independientemente** sus top-$k$ tokens por afinidad, **relajando la partición dura en celdas suaves superpuestas**. Un token puede pertenecer a más de una burbuja → desaparece el conflicto.

[ARCH] Alternativa equivalente: **Latent Prototype Routing** (asignación a centroides-prototipo con balance controlable; reduce el Gini de asignación de 0.70 a 0.035) — casi exactamente el setting de centroides del BT. Tercera vía: **Power Diagram suave** (softmax sobre power distances) para que routing y geometría coincidan por construcción.

[ARCH] **Decisión V5:** partición soft por defecto (MoSA), con opción de colapsar a partición dura (Straight-Through, §6) en inferencia para recuperar la sparsity exacta de bloque. Entrenamiento suave, inferencia dura.

---

## §9 — Capas híbridas [NEW, del survey]

[NEW] **La lección universal de 2025:** ningún mecanismo sub-cuadrático puro retiene capacidad de retrieval de largo alcance. Híbridos de producción (Qwen3-Next, Kimi Linear, Jamba) interleavean capas eficientes con unas pocas de atención completa o MLA.

[ARCH] **BT V5 híbrido:** schedule de capas con ratio configurable (default 3:1) —
$$[\underbrace{\text{Bubble}}_{\times 3},\ \underbrace{\text{Full-Attn / MLA}}_{\times 1}]\ \text{repetido}.$$
Las capas de atención completa (o MLA comprimida) actúan como "puentes globales" que recuperan dependencias inter-burbuja de largo alcance que el routing geométrico no captura. Las capas Bubble cargan el grueso del cómputo a O(N·C).

⚠ Esto es una **concesión estratégica**: el BT puro era la apuesta elegante; el BT híbrido es la apuesta competitiva. La evidencia 2025 dice que el híbrido gana. Decidir según el benchmark RULER/NIAH (Paper V).

---

## §10 — Resumen arquitectónico: V4 → V5

| Componente | V4 | V5 | Naturaleza |
|---|---|---|---|
| RoPE | ausente (bug) | integrado pre-clustering | [FIX] |
| Asignación token→burbuja | argmin duro (no diff.) | Gumbel-Sinkhorn + ST | [FIX] Blocker 1 |
| Expert-Choice × Power Diagram | incompatible | MoSA soft / Latent Prototype | [FIX] Blocker 2 |
| Nº de burbujas C | fijo | Baroreceptor MLP dinámico | [NEW] |
| Kernel Sinkhorn | log-domain naive | FlashSinkhorn (IO-aware) | [NEW] (Paper IV) |
| Topología de capas | homogénea (todo Bubble) | híbrida 3:1 con Full/MLA | [NEW] |
| Geometría | Laguerre + 4 manifolds | igual | conservado |
| Fundamento | ε→0 = Power Diagram (Litman) | igual + Γ-convergencia formal | conservado |

[ARCH] **El núcleo conservado:** el BT sigue siendo "atención = límite $\varepsilon\to 0$ del OT entrópico one-sided, computado como teselación de Laguerre en un paso". Eso no cambia entre V4 y V5. Lo que cambia es que V5 **entrena de punta a punta** (Gumbel-Sinkhorn), **rutea sin conflicto** (MoSA), **predice su propia granularidad** (Baroreceptor), y **compite** (híbrido + FlashSinkhorn kernel).

---

*Siguiente: Paper II — SIRI: teoría del effective rank y el flanco de defensa ante reviewers.*
