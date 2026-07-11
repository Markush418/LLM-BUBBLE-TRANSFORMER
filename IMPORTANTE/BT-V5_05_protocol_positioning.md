# BT V5 · Paper V — Protocolo Experimental y Posicionamiento
### Benchmarks, thresholds de decisión, related work corregido, y checklist arXiv

> **Qué es esto.** El plan de validación del BT V5 y su posicionamiento honesto frente al estado del arte. Define qué medir, qué umbrales cambian la estrategia, cómo citar correctamente (la atribución Litman≠Daneshmand es motivo de rechazo si está mal), cuál es la novedad defendible, y la checklist concreta para llegar a arXiv.
>
> **Tags:** `[BENCH]` benchmark · `[THRESHOLD]` umbral de decisión · `[CITE]` corrección bibliográfica · `[NOVELTY]` · `⚠` · `☐` acción.

---

## §1 — El blocker crítico: ΔPPL

[THRESHOLD] **El gate que manda sobre todos los demás.** Antes de cualquier claim de speedup o de SIRI, el BT V5 debe demostrar sobre Qwen3-0.6B con SDOT inyectado (y RoPE ya integrado):
$$\Delta\text{PPL}=\text{PPL}_{\text{BT}}-\text{PPL}_{\text{softmax}}\le \delta_{\text{threshold}}\quad(\delta\le 2\%).$$

⚠ **Las métricas internas NO sustituyen a la PPL.** $R_{\text{eff}}$, concentration ratio, anisotropy son métricas geométricas — válidas para SIRI, **inválidas** como prueba de capacidad predictiva. Un BT con $R_{\text{eff}}$ espectacular y ΔPPL = +50% es un fracaso. La PPL es el árbitro.

[BENCH] **Protocolo:** re-correr `perplexity_benchmark.py` con RoPE integrado (post-fix del incidente 831,974), sobre el mismo corpus, comparando softmax baseline (PPL ≈ 20.03) vs BT V5.

---

## §2 — Suite de benchmarks

[BENCH] **Tres ejes, cada uno con su rival natural:**

| Eje | Benchmark | Mide | Rival a vencer |
|---|---|---|---|
| Calidad de lenguaje | PPL (WikiText/corpus host) | capacidad predictiva | softmax baseline |
| Contexto largo / retrieval | RULER, Needle-in-a-Haystack (64k) | recuperación de largo alcance | híbridos lineales (Kimi/GDN) |
| Eficiencia estructural | Long Range Arena (LRA) | accuracy a complejidad sub-cuadrática | LOTFormer, Routing Transformer |
| Velocidad | wall-clock @ N∈{8K,32K,128K} | speedup real (no FLOPs) | LOTFormer, FlashAttention |

[BENCH] **Métricas de validación de conjeturas (Sprint 3-4):**
- Distribución real de tokens por burbuja → valida/invalida el supuesto de balance del speedup (Paper IV §2).
- $\beta_0(G_\varepsilon),\beta_1(G_\varepsilon)$ durante el training → valida la conjetura de phase transitions (Paper II §6).
- Reproducibilidad del pico SIRI a través de normalizaciones de costo, tamaños de modelo, y capas (Paper II §7).

---

## §3 — Thresholds que cambian la estrategia

[THRESHOLD] Decisiones binarias pre-comprometidas (evita el sesgo de confirmación post-hoc):

1. **Si ΔPPL > 2%** → el BT puro no preserva calidad → activar el **BT híbrido** (Paper I §9: interleaving con Full/MLA). Si el híbrido tampoco → revisar el fix de RoPE o el routing.
2. **Si el BT no vence a LOTFormer en LRA accuracy *y* wall-clock** → abandonar el solver OT a medida y **construir sobre la formulación de pivotes de LOTFormer**. (El survey es explícito: LOTFormer ya publicó lo que el BT intenta; sin superioridad medible, el BT no tiene caso.)
3. **Si la PPL del BT traila a un Gated DeltaNet del mismo tamaño por >5–10%** → pivotear a BT híbrido en vez de bubble puro.
4. **Si el pico SIRI no es reproducible** → reformular SIRI como fenómeno **model-specific** de Qwen3-0.6B, no como ley general. (No invalida el paper, pero cambia el claim.)
5. **Si la distribución de tokens por burbuja es muy sesgada** → el speedup teórico no se materializa → reportar el speedup empírico real, no el de FLOPs.

[ARCH] Comprometerse con estos thresholds **antes** de correr los experimentos es disciplina anti-vibecoding: el resultado decide la arquitectura, no al revés.

---

## §4 — Related Work: atribuciones corregidas [CITE — crítico]

⚠ **Un error de atribución en arXiv es motivo de rechazo inmediato.** Correcciones verificadas:

[CITE] **Error 1 — el fundamento teórico central NO es Daneshmand:**
```
arXiv:2508.08369
Título: "Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport"
Autor: Elon Litman (2025)   ← NO "Daneshmand et al."
```
Este paper prueba que el forward pass de SDPA es la solución exacta del OT entrópico **one-sided** → es el fundamento directo de la convergencia del BT al Power Diagram. Citarlo mal hunde el paper.

[CITE] **Error 2 — Daneshmand SÍ existe pero es otro alcance:**
```
arXiv:2410.19931
Título: "Provable optimal transport with transformers: the essence of depth and prompt engineering"
Autor: Hadi Daneshmand et al. (2024)
```
Prueba que las capas de self-attention simulan gradient descent sobre el **dual** del OT entrópico, donde la **profundidad** controla la precisión. Es un resultado **distinto** al que el BT necesita. Diferenciar en el abstract: [1] Daneshmand = OT vía profundidad; [2] Litman = OT one-sided, solución exacta.

[CITE] **Papers no citados que un reviewer exigirá:**

| ID | Título | Por qué es obligatorio |
|---|---|---|
| arXiv:2509.23436 | LOTFormer | ⚡ competidor directo — su ausencia es un gap fatal |
| arXiv:2202.09368 | Expert-Choice Routing (Zhou 2022) | base del routing V4/V5 |
| arXiv:2505.00315 | MoSA | el fix del Blocker 2 |
| arXiv:1802.08665 | Gumbel-Sinkhorn (Mena 2018) | el fix del Blocker 1 |
| arXiv:2601.15380 | Better Attention Priors | EOT priors → Baroreceptor |
| arXiv:2601.19942 | Latent Object Permanence (phase transitions) | valida conjetura del Baroreceptor |
| arXiv:2410.11042 | Persistent Topological Features in LLMs | base de la Meseta topológica (Betti) |
| arXiv:2602.03067 | FlashSinkhorn | kernel IO-aware |

⚠ **IDs con fecha futura a verificar antes de citar formalmente:** FlashSinkhorn (2602.03067), Mamba-3 (2603.15569), CARE (2603.17946), k-MIP (2604.03815), Sinkhorn rank decay (2604.07925), y los IDs 2601.* de phase transitions. Confirmar existencia en arXiv antes de la submission (algunos surgieron solo vía búsqueda y no se verificaron independientemente).

---

## §5 — La novedad defendible del BT [NOVELTY]

[NOVELTY] Tras el survey, lo que el BT retiene como contribución genuina (no cubierta por la competencia):

1. **SIRI** — el fenómeno empírico de inflación de rango por sparsity, con su conjetura mecanística (ortogonalización forzada por presión de reconstrucción) y el sketch hacia un teorema vía el dual one-sided de Litman. **Ningún paper de la competencia reporta esto.** Es la contribución titular.
2. **La Meseta de Saturación topológica** — definir el régimen de operación óptimo vía invariantes de Betti ($\beta_0,\beta_1$ constantes), independiente de precisión numérica. Definición novel.
3. **El Power Diagram como límite exacto** (no aproximación) de softmax en $\varepsilon\to 0$, computado en un paso sin iteraciones. LOTFormer aproxima con garantías; el BT afirma ser el límite exacto. (⚠ Este claim necesita el respaldo formal de la Γ-convergencia, Documento 0 Capa F, y resistir la comparación de velocidad real.)
4. **El Baroreceptor MLP** como detector de régimen líquido/sólido que ajusta $C$ dinámicamente — conexión novel entre routing geométrico y la teoría de phase transitions.

[ARCH] **Lo que el BT NO puede reclamar como novel:** la atención por OT (Sinkformers, 2022), la atención por centroides (Routing Transformer, 2021), la atención doblemente estocástica linear-time (LOTFormer, 2025), Expert-Choice (Zhou, 2022). Todo eso es prior art. La honestidad sobre qué es y qué no es novel es lo que distingue un paper aceptable de uno rechazado por overclaiming.

---

## §6 — Checklist arXiv (Sprint 3+)

```
ERRORES A CORREGIR (pre-submission, no negociables):
☐ Atribución: "Daneshmand" → "Litman, E." para arXiv:2508.08369
☐ Diferenciar claims [1] Daneshmand (profundidad) vs [2] Litman (one-sided exacto)
☐ Speedup: presentar par (124× atención, ~54% modelo) — nunca el 124× solo
☐ S_sat: reemplazar threshold 10⁻⁵ por definición topológica (Betti) o justificarlo
☐ R_eff: especificar que se mide sobre X (embeddings), no sobre A (attention map)
☐ Desambiguar SIRI: rango espectral de X, no sparsity del soporte de A (Paper II §4)

PAPERS A AGREGAR (§4):
☐ LOTFormer + tabla de diferenciación (Related Work)
☐ MoSA, Gumbel-Sinkhorn, Expert-Choice (sección de métodos)
☐ Better Attention Priors, phase transitions ×3 (Baroreceptor + Meseta)
☐ Verificar todos los IDs con fecha 2026 antes de citar

EXPERIMENTOS (Sprint 3 — gate crítico primero):
☐ ΔPPL ≤ 2% sobre Qwen3-0.6B con RoPE  ← BLOCKER, hace todo lo demás
☐ Cerrar Blockers 1 y 2 (Gumbel-Sinkhorn + MoSA) e integrarlos
☐ Wall-clock real @ N∈{8K,32K,128K} vs LOTFormer + FlashAttention
☐ LRA accuracy vs LOTFormer y Routing Transformer
☐ RULER/NIAH vs híbrido lineal (decide BT puro vs BT híbrido)
☐ Distribución de tokens por burbuja (valida supuesto de balance)
☐ β₀,β₁ de G_ε durante training (valida phase transitions)
☐ Reproducibilidad del pico SIRI (3 ejes: costo, tamaño, capa)

FORMALIZACIÓN (eleva el paper, opcional pero alto valor):
☐ Cerrar el SKETCH de SIRI: ortogonalización vía dual one-sided (Litman)
   → convierte SIRI de [CONJECTURE] a [THEOREM] ← la pieza de mayor valor
☐ Formalizar la Meseta topológica con gudhi (implementación + prueba de invariancia)
```

---

## §7 — Orden de ejecución recomendado

[ARCH] La secuencia que minimiza riesgo y maximiza señal:

```
1. ΔPPL con RoPE          → ¿el BT siquiera funciona? (gate absoluto)
2. Cerrar Blockers 1+2    → ¿entrena end-to-end?
3. Wall-clock vs LOTFormer → ¿es competitivo en velocidad real?
4. LRA + RULER            → ¿BT puro o híbrido?
5. Reproducibilidad SIRI  → ¿la contribución titular es robusta?
6. Cerrar sketch SIRI     → ¿[CONJECTURE] → [THEOREM]?
7. Submission arXiv       → con related work corregido y claims honestos
```

[NOVELTY] **La tesis del paper en una frase, post-todo-este-chat:** *"El Bubble Transformer computa atención como el límite exacto $\varepsilon\to 0$ del transporte óptimo entrópico one-sided —una teselación de Laguerre en un solo paso, sin las iteraciones de Sinkhorn— y al hacerlo revela SIRI: un fenómeno empírico de inflación del effective rank bajo sparsity, contrario a la predicción de rank collapse de la literatura, con un mecanismo conjetural de ortogonalización forzada que el dual one-sided de Litman puede formalizar."*

⚠ Esa frase es defendible **solo si** el gate ΔPPL pasa y SIRI es reproducible. Todo lo demás es teoría elegante sin evidencia. El orden importa: primero el gate, después la elegancia.

---

*Fin de la suite BT V5. Documentos: [0] Bases primarias · [I] Arquitectura · [II] SIRI · [III] Blockers · [IV] Complejidad/Scaling · [V] Protocolo/Posicionamiento.*
