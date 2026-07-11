# BT V5 · Paper III — Resolución de Blockers
### Diferenciabilidad (argmin), routing (Expert-Choice × Power Diagram), y RoPE

> **Qué es esto.** El tratamiento formal de los tres puntos que bloqueaban a V4, cada uno con su análisis matemático, su fix verificado en la literatura, y los constraints de implementación. Cerrar estos tres es lo que convierte V4 (prototipo con blockers) en V5 (entrenable end-to-end).
>
> **Tags:** `[PROBLEM]` · `[FIX]` · `[THM]` · `[GRAD]` análisis de gradiente · `[IMPL]` constraint de implementación · `⚠`.

---

## §1 — Blocker 1: no-diferenciabilidad del argmin

### 1.1 El problema

[PROBLEM] La asignación token→burbuja es
$$b(x)=\arg\min_i\ \operatorname{pow}(x,c_i),\qquad \operatorname{pow}(x,c_i)=\|x-c_i\|^2-\psi_i.$$
El $\arg\min$ produce un índice discreto → la salida es una **función escalonada** de $(c_i,\psi_i)$: gradiente **cero** casi en todas partes, **indefinido** en las fronteras de celda. Backprop no propaga señal → los centroides $c_i$ y los pesos $\psi_i$ **no se aprenden**.

### 1.2 Fix A — Gumbel-Sinkhorn (elegido)

[FIX] *(Mena et al. 2018, arXiv:1802.08665)* Reemplazar el argmin duro por una matriz de asignación soft, doblemente estocástica, construida con Sinkhorn sobre logits perturbados con ruido Gumbel:
$$\Pi_\tau=\operatorname{Sinkhorn}\Big(\frac{-\operatorname{pow}(x,c)+g}{\tau}\Big),\qquad g_{ij}\sim\operatorname{Gumbel}(0,1)\ \text{i.i.d.}$$

[THM] *(Convergencia al límite duro)* Cuando $\tau\to 0$, $\Pi_\tau$ converge en distribución a la asignación de matching duro (la permutación de bloque óptima). Para $\tau>0$, $\Pi_\tau$ es **diferenciable** respecto de $c_i,\psi_i$ (todo el operador Sinkhorn es composición de exp, divisiones y sumas — suave).

[GRAD] El gradiente fluye a través de las iteraciones de Sinkhorn (cada una diferenciable). El ruido Gumbel hace la relajación insesgada respecto del muestreo de la asignación dura (Gumbel-Max trick, Documento de estudio §6).

[ARCH] **Por qué este y no Gumbel-Softmax plano.** La asignación tiene estructura de **acoplamiento con restricción de capacidad** (cada burbuja recibe a lo sumo $\sim N/C$ tokens), no de elección categórica independiente. Gumbel-Softmax plano ignora la restricción de capacidad; Gumbel-Sinkhorn la respeta usando **el mismo Sinkhorn que el BT ya tiene**. Cero maquinaria nueva, consistencia con el framework OT.

### 1.3 Fix B — MESH (alternativa)

[FIX] **MESH (Minimize Entropy of Sinkhorn):** aumentar la matriz de costo de modo que la solución entrópica sea **casi one-hot** (efectivamente dura) manteniéndose diferenciable. Útil si Gumbel-Sinkhorn introduce demasiada varianza de gradiente.

### 1.4 Straight-Through: cómputo duro, gradiente suave

[FIX] **Straight-Through estimator.** Forward: usar la asignación **dura** $\arg\max_i\Pi_{\tau}$ (preserva la complejidad O(N·C) y la sparsity de bloque exacta). Backward: usar el gradiente de $\Pi_\tau$ **soft**. Implementación canónica:
```
Π_hard = onehot(argmax(Π_soft))
Π = Π_hard + Π_soft − stop_gradient(Π_soft)   # forward: Π_hard ; backward: ∇Π_soft
```

[GRAD] **Validación requerida:** benchmarkear varianza del gradiente y ΔPPL de Gumbel-Sinkhorn+ST vs un baseline straight-through puro vs MESH. La métrica de decisión es estabilidad de entrenamiento, no elegancia.

---

## §2 — Blocker 2: Expert-Choice × Power Diagram

### 2.1 El problema

[PROBLEM] **Expert-Choice routing** (Zhou et al. 2022, arXiv:2202.09368): invierte token-choice → los **expertos eligen** sus top-$k$ tokens → balance de carga perfecto, sin auxiliary loss. **Power Diagram duro:** cada token cae en **exactamente una** celda → es token-choice de **capacidad 1**. Estos dos esquemas son **estructuralmente incompatibles**: Expert-Choice necesita que un experto pueda tomar muchos tokens y que un token pueda ir a varios expertos; la partición dura prohíbe ambas cosas.

### 2.2 Fix — MoSA: "los expertos eligen tokens"

[FIX] *(Mixture of Sparse Attention, arXiv:2505.00315)* Cada burbuja/experto $i$ selecciona **independientemente** sus top-$k$ tokens por afinidad. Esto **relaja la partición dura en celdas suaves superpuestas**: un token puede pertenecer a varias burbujas → el conflicto desaparece por construcción.

[THM] Complejidad por head ~$O(k^2+T)$ (MoSA), compatible con el objetivo O(N·C) del BT cuando $k\sim N/C$.

### 2.3 Alternativas equivalentes

[FIX] **Latent Prototype Routing:** generaliza Expert-Choice como asignación a centroides-prototipo aprendidos con balance controlable. Reduce el coeficiente de Gini de asignación de **0.70 → 0.035** — casi exactamente el setting de centroides del BT. Es Expert-Choice nativo sobre prototipos = burbujas.

[FIX] **Power Diagram suave:** definir la asignación como $\operatorname{softmax}_i(-\operatorname{pow}(x,c_i)/\tau)$ en vez del argmin. Routing y geometría **coinciden por construcción** (no hay dos mecanismos que reconciliar). Es el camino más limpio teóricamente; el costo es perder la sparsity exacta de bloque en entrenamiento.

### 2.4 Decisión V5

[ARCH] **Entrenamiento suave (MoSA / Power Diagram suave), inferencia dura (Straight-Through colapsa a partición exacta).** Esto resuelve el blocker sin sacrificar la sparsity de bloque que da el speedup: se entrena con celdas superpuestas diferenciables, se infiere con la teselación dura O(N·C).

---

## §3 — Blocker 3: RoPE en atención no estándar

### 3.1 El incidente y su diagnóstico

[PROBLEM] El primer run de `perplexity_benchmark.py` dio:
```
PPL softmax baseline:   20.03
PPL bubble (ε=0.005):   831,974.76     → regresión catastrófica
```
[THM] **Causa raíz:** `SDOTAttentionV4` no aplicaba RoPE. Diagrama causal:
```
sin RoPE → Q,K sin rotación posicional
        → routing de burbujas basado SOLO en contenido semántico
        → clusters geométricos posicionalmente incoherentes
        → attention patterns inconsistentes con el LM head
        → logits incorrectos → NLL altísimo → PPL = 831,974
```
El smoke test pasó igual porque solo verifica shapes (`[1,64,1024]→[1,64,1024]`), no corrección numérica.

### 3.2 El fix y su orden

[FIX] Integrar `rotary_emb` y `position_ids` en el módulo de atención del BT. **RoPE se aplica post-proyección, pre-clustering:**
$$Q\leftarrow R_\Theta(\text{pos})\,Q,\quad K\leftarrow R_\Theta(\text{pos})\,K\quad\text{(antes del routing de burbujas)}.$$

[THM] **Por qué ese orden es obligatorio.** El routing geométrico opera en el espacio $Q$-$K$. Si $Q,K$ no llevan posición, los centroides aprenden clusters solo semánticos → burbujas posicionalmente incoherentes. RoPE **antes** del clustering hace que las burbujas sean coherentes en posición+contenido.

### 3.3 Constraints de implementación (Qwen3, transformers 4.51.0)

[IMPL]
| Constraint | Razón |
|---|---|
| `rotary_emb=None` como default | backward-compat: tests sin `position_ids` siguen pasando |
| RoPE **después** de proyección, **antes** de clustering | burbujas posicionalmente coherentes |
| Firma `rotary_emb(x, position_ids)` (no solo `position_ids`) | Qwen3 necesita `x` para inferir dtype/device |
| Return `(output, None)` — exactamente 2 valores | `DecoderLayer` desempaqueta 2; 3 crashea en silencio |
| Aplicar RoPE **antes** de la expansión GQA | Qwen3 lo hace pre-expansion; seguir el mismo orden |

### 3.4 Lección arquitectónica general

⚡ **Cualquier módulo que reemplace `self_attn` en un Transformer posicional DEBE heredar o reimplementar el encoding posicional del host.** Esta es la regla que el incidente cristalizó, y aplica a todo mecanismo sub-cuadrático (no solo al BT).

### 3.5 Avances 2025-2026 en positional encoding (opcionales para V6)

[FIX] Para versiones futuras, alternativas a RoPE estándar que encajan con atención clusterizada:
- **Selective RoPE** (arXiv:2511.17388): ángulos dependientes del input, para atención lineal *y* softmax.
- **DoPE** (denoising RoPE, arXiv:2511.09146).
- **Mamba-3** (arXiv:2603.15569): expresa posición vía transiciones de estado complejas (= rotaciones) → plantilla para codificar posición en la **matriz de costo** del OT en lugar de en $Q,K$.
- **MLA pattern** (DeepSeek): aislar RoPE en un camino dedicado, separado del latente comprimido.

[ARCH] **Opción V6 elegante:** en vez de rotar $Q,K$, **codificar la posición en la matriz de costo** $C_{ij}$ del OT (penalizar transporte entre posiciones distantes). Esto hace el positional encoding **nativo del mecanismo de transporte** en lugar de un add-on. No verificado; candidato de investigación.

---

## §4 — Checklist de cierre de blockers (Sprint 3)

```
BLOCKER 1 (diferenciabilidad):
☐ Implementar Gumbel-Sinkhorn para asignación token→burbuja
☐ Añadir Straight-Through (forward duro / backward suave)
☐ Benchmarkear varianza de gradiente: GS+ST vs MESH vs baseline
☐ Verificar que ψ_i y c_i reciben gradiente no nulo

BLOCKER 2 (routing):
☐ Implementar MoSA (la burbuja elige top-k tokens)
☐ Validar balance de carga (medir Gini de asignación, target <0.05)
☐ Confirmar entrenamiento-suave → inferencia-dura preserva sparsity

BLOCKER 3 (RoPE):  ✅ FIXED (run pendiente de re-medición)
☐ Re-correr perplexity_benchmark con RoPE integrado
☐ Confirmar ΔPPL ≤ 2% (blocker crítico de Sprint 3, Paper V)
☐ Verificar orden: RoPE post-proyección, pre-clustering, pre-GQA-expansion
```

[ARCH] **Estado:** Blocker 3 ya tiene el fix implementado (falta re-medir). Blockers 1 y 2 tienen fix **identificado y verificado en la literatura** — falta implementar. Ninguno es un problema de investigación abierto; los tres son ingeniería con respaldo teórico.

---

*Siguiente: Paper IV — Complejidad, speedup honesto, y scaling a modelos grandes.*
