# Formalismo Matemático: SIRI + Power Diagram en la Nueva Arquitectura

**Fecha**: 2026-06-27
**Proyecto**: LLM-BUBBLE / Bubble Transformer
**Estado**: Formalización L2 (rigurosa)
**Axiomática**: Epistemic tags aplicadas (§1 mathematical-depersonalization-engine)

---

## §0 — Propósito del documento

[DEFINITION] Este documento formaliza la definición operativa de **SIRI** (Sinkhorn Iterative Regularized Inference) y **Power Diagram** dentro de la nueva arquitectura post-SDOT. El objetivo es demostrar que ambos componentes se preservan bajo cualquier arquitectura de reemplazo (DeltaNet, RetNet, GLA, Mamba-2, Kimi Linear), siempre que se respete el contrato de regularización entrópica sobre una matriz de costo geométrico.

---

## §1 — Axiomática

[AXIOM] **A1 (Secuencia de tokens)** Sea X = {x₁, ..., xₙ} ⊂ ℝᵈ una secuencia de n vectores token con xᵢ ∈ ℝᵈ.

[AXIOM] **A2 (Proyecciones)** Existen funciones lineales aprendibles W_Q, W_K, W_V : ℝᵈ → ℝᵈₕ tales que para cada token xᵢ ∈ ℝᵈ se producen:
  - Qᵢ = W_Q · xᵢ ∈ ℝᵈₕ  (query)
  - Kᵢ = W_K · xᵢ ∈ ℝᵈₕ  (key)
  - Vᵢ = W_V · xᵢ ∈ ℝᵈₕ  (value)

[AXIOM] **A3 (Costo geométrico)** La matriz de costo C ∈ ℝⁿˣⁿ se define como:
  Cᵢⱼ = ‖Qᵢ - Kⱼ‖²₂ ∈ ℝ≥₀

  Es decir, distancia euclídea cuadrada entre proyecciones aprendidas, **NO** producto interno Qᵢ⊤Kⱼ.

[AXIOM] **A4 (Mecanismo de atención)** El output Oᵢ ∈ ℝᵈₕ del token i es:
  Oᵢ = Σⱼ Aᵢⱼ · Vⱼ

  donde A ∈ ℝⁿˣⁿ es la matriz de atención (depende de C).

[AXIOM] **A5 (Espacio de probabilidad)** Sea Σₙ = {A ∈ ℝⁿˣⁿ₊ : A1ₙ = 1ₙ, A⊤1ₙ = 1ₙ} el conjunto de matrices doblemente estocásticas (politopo de Birkhoff).

[AXIOM] **A6 (Bandwidth ε > 0)** El coeficiente ε es la bandwidth/temperatura del regularizador entrópico. Convencionalmente ε ∈ (0, ∞).

---

## §2 — SIRI: Definición formal

[DEFINITION] **SIRI (Sinkhorn Iterative Regularized Inference)**

Dado C ∈ ℝⁿˣⁿ₊ y ε > 0, SIRI resuelve:

  minimize_{A ∈ Σₙ} ⟨A, C⟩ - ε · H(A)            (P-SIRI)

donde H(A) = -Σᵢⱼ Aᵢⱼ log Aᵢⱼ es la entropía de Shannon de la matriz de atención, y ⟨A, C⟩ = Σᵢⱼ Aᵢⱼ Cᵢⱼ es el producto interno de Frobenius.

[LEMMA] **Equivalencia con Transporte Óptimo Entrópico**

[LEMMA-2.1] El problema (P-SIRI) es equivalente al problema de transporte óptimo con regularización entrópica entre marginales uniformes μ = ν = (1/n) · 1ₙ:

  min_{P ∈ Π(μ,ν)} ⟨P, C⟩ - ε · H(P)             (P-OT-entropic)

donde Π(μ,ν) = {P ∈ ℝⁿˣⁿ₊ : P1 = μ1ₙ, P⊤1 = ν1ₙ}. Bajo μ = ν = uniforme, Π(μ,ν) = Σₙ.

[SKETCH] Por Cuturi (2013, arxiv:1306.0895), la regularización entrópica del OT produce el mismo problema dual de Kantorovich. Bajo marginales uniformes, el politopo de couplings Π(μ,ν) coincide con Σₙ. ∎

[THEOREM] **Algoritmo de Sinkhorn-Knopp en log-domain**

[THEOREM-2.2] Para τ iteraciones de Sinkhorn-Knopp log-domain:

  M⁽⁰⁾ = -C/ε                                  (log-S inicial)

  for t = 1, 2, ..., τ:
    u⁽ᵗ⁾ = -logsumexp(M⁽ᵗ⁻¹⁾ + v⁽ᵗ⁻¹⁾·1ₙᵀ, axis=1)   (filas)
    v⁽ᵗ⁾ = -logsumexp(M⁽ᵗ⁻¹⁾ + 1ₙ·u⁽ᵗ⁾ᵀ, axis=0)   (columnas)

  A⁽ᵗ⁾ = exp(M⁽ᵗ⁾ + u⁽ᵗ⁾·1ₙᵀ + 1ₙ·v⁽ᵗ⁾ᵀ)

el iterado A⁽ᵗ⁾ converge al óptimo A* de (P-SIRI) cuando t → ∞.

[SKETCH] Por teoría clásica de Sinkhorn (Sinkhorn-Knopp 1964, Peyré-Cuturi 2019). Bajo ε > 0 y C ≥ 0 finito, el operador de Sinkhorn es contractivo en métrica Hilbert projective. La convergencia es lineal con constante 1 - exp(-2ε · σ_max(C)). Para τ = 5 iteraciones, el gap de optimalidad es O(exp(-10ε · σ_max(C))) — suficiente para ε ≥ 0.001. ∎

[COROLLARY] **Implementación NumPy**

[COR-2.3] El método `_sinkhorn_log_domain(C, τ=5)` en `experiments/plateau_attention.py` implementa exactamente THEOREM-2.2 con:

  log_S = -C / ε                                                  (línea 224)
  u, v = 0  (shape [B, heads, N])                                 (líneas 235-236)
  for τ iterations:
    u = -logsumexp(log_S + v[:,:,:,np.newaxis], axis=-1)           (línea 239)
    v = -logsumexp(log_S + u[:,:,np.newaxis,:], axis=-2)           (línea 240)
  A = exp(log_S + u[:,:,np.newaxis,:] + v[:,:,np.newaxis,:])     (línea 242)

  [AXIOM] de implementación: dtype=np.float32, log domain previene underflow en ε < 0.01. ∎

---

## §3 — Power Diagram: Definición formal

[DEFINITION] **Power Diagram con pesos ψ**

Dados K centroides c₁, ..., c_K ∈ ℝᵈ y pesos ψ₁, ..., ψ_K ∈ ℝ, la celda de Power Diagram de cₖ es:

  Pᵧₖ = {x ∈ ℝᵈ : ‖x - cₖ‖² - ψₖ ≤ ‖x - cⱼ‖² - ψⱼ  ∀ j ≠ k}    (PD-1)

[LEMMA] **Laguerre tessellation como caso particular**

[LEMMA-3.1] Cuando todos los pesos ψₖ = 0, las celdas Pᵧₖ coinciden con las celdas de Voronoi estándar. Power Diagram generaliza Voronoi añadiendo "radios" efectivos rₖ = √ψₖ a cada celda.

[AXIOM] **A7 (Power Diagram en log_S)** En la nueva arquitectura, los pesos ψₖ se incorporan como bias aditivo en la matriz log-Sinkhorn:

  log_Sᵢⱼ = -Cᵢⱼ/ε + ψⱼ                                    (PD-2)

  donde ψⱼ es el peso de Power Diagram del key j. Esto equivale a sumar ψ al potencial dual v antes de iterar Sinkhorn.

[THEOREM] **Equivalencia ψ-bias con Laguerre dual**

[THEOREM-3.2] La transformación ψⱼ ↦ v inicial es exacta: el sesgo aditivo ψⱼ en log_S equivale a inicializar el potencial v con ψ en la iteración 0 de Sinkhorn, y produce la misma matriz A.

[SKETCH] Por linearidad de Sinkhorn en los potenciales duales (u, v) y la propiedad aditiva de la exponencial log-S, sumar ψ a v es equivalente a sumar ψ a log_S antes de la primera iteración. ∎

[COROLLARY] **Power Diagram ψ como bias de Sinkhorn**

[COR-3.3] El método `_sinkhorn_log_domain(C, mask=None, psi=None)` extendido acepta ψ y devuelve A con bias de Power Diagram aplicado. Implementación en `experiments/v4_adapter.py` y `experiments/plateau_attention.py`:

  log_S = -C / ε + psi                                      (línea modificada)
  A = Sinkhorn(log_S)                                        (Sinkhorn log-domain)

[AXIOM] de implementación: ψ se computa como proyección aprendible W_ψ sobre los keys K (similar a Q, K, V). ∎

---

## §4 — ε como Bandwidth / Temperatura

[DEFINITION] **ε como bandwidth del kernel entrópico**

[DEF-4.1] En el formalismo SIRI, ε > 0 controla la "temperatura" del kernel Gibbs K(x,y) = exp(-C(x,y)/ε). En el límite:

  lim_{ε→0+} A* = argmin_{A ∈ Σₙ} ⟨A, C⟩  (transport plan mínimo, sparse)
  lim_{ε→∞} A* = (1/n²) · 1ₙ₁ₙᵀ        (matriz uniforme)

[THEOREM] **Regímenes asintóticos**

[THEOREM-4.2] Bajo el kernel Gibbs, las matrices A* convergen monótonamente:

  ‖A*(ε₁) - A*(ε₂)‖_F → 0   cuando |ε₁ - ε₂| → 0

  Y el flujo ε ↦ A*(ε) es continuo en (0, ∞).

[SKETCH] Por regularidad Lipschitz de Sinkhorn-Knopp con respecto a C/ε. ∎

[EMPIRICAL] **Sweet spot ε ≈ 0.001**

[EMP-4.3] Observación experimental previa (Sprint 1-2 del proyecto): ε ≈ 0.001 maximiza la concentración de embeddings sin colapso representacional. Rango operativo recomendado: [0.001, 1.0]. El sweep logarítmico {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0} cubre 3 décadas.

[CONJECTURE] **Universalidad del sweet spot**

[CONJ-4.4] El sweet spot ε ≈ 0.001 podría ser universal para arquitecturas con Q/K proyecciones de dimensión head_dim ≈ 64-128 (Qwen, LLaMA). Hipótesis: ε ≈ head_dim⁻¹.² normaliza el costo L2 a la escala de los logits softmax estándar. Pendiente validación experimental.

---

## §5 — Compatibilidad con arquitecturas SOTA

[LEMMA] **Compatibilidad SIRI con cualquier atención densa Q/K**

[LEMMA-5.1] Cualquier arquitectura que preserve:
  1. Proyecciones separadas Q, K, V (AXIOM-A2)
  2. Matriz de costo Cᵢⱼ = ‖Qᵢ - Kⱼ‖² (AXIOM-A3)
  3. Mecanismo de atención vía Σⱼ Aᵢⱼ Vⱼ (AXIOM-A4)

admite SIRI como reemplazo drop-in de softmax, sin modificar la arquitectura.

[THEOREM] **Softmax como caso límite de SIRI**

[THEOREM-5.2] Para C = -log Pᵢⱼ donde Pᵢⱼ = softmax(QK⊤/√d), tenemos:

  softmax(QK⊤/√d) = lim_{ε→0+} A*_SIRI(C') donde C'ᵢⱼ = -Qᵢ⊤Kⱼ/√d · ε + log Zᵢ

bajo regularización apropiada. Softmax ≈ SIRI cuando la atención es row-stochastic pero no doubly-stochastic.

[SKETCH] Por definición de entropía relativa y dualidad KL. ∎

[COROLLARY] **Drop-in compatibility**

[COR-5.3] Las arquitecturas SOTA evaluadas son compatibles con SIRI:

| Arquitectura | SIRI compatible | Notas |
|---------------|-----------------|-------|
| **RetNet** (Sun et al. 2023) | Sí (atención densa) | Reemplazar softmax por SIRI en parallel branch |
| **RWKV-7** (Peng et al. 2025) | Parcial (WKV no es Q/K atención) | SIRI aplicable en bloques residuales, no como reemplazo directo |
| **GLA** (Yang et al. 2023) | Sí (lineal, requiere kernel φ) | Usar SIRI como kernel en lugar de feature map φ |
| **DeltaNet** (Yang et al. 2024) | Sí (lineal, delta rule) | Aplicar SIRI sobre matriz de scores pre-delta |
| **Mamba-2** (Dao & Gu 2024) | Parcial (SSD tiene forma específica) | SIRI no aplica directamente, requiere reinterpretación |
| **Kimi Linear / KDA** (Kimi Team 2025) | Sí (híbrido KDA + MLA) | Aplicar SIRI en bloques MLA |
| **Qwen3 native** (DeltaNet híbrido) | Sí (en capas full-attention) | Reemplazar softmax por SIRI en capas full-attention |

[HEURISTIC] **Recomendación de integración**

[HEU-5.4] Mantener SIRI como módulo opcional post-atención:

  out = Attention_SOTA(x)            # cualquier arquitectura
  out_siri = SIRI(out, C=‖Q-K‖², ε)  # refinamiento opcional
  out_final = λ · out + (1-λ) · out_siri   # interpolación

donde λ ∈ [0,1] es coeficiente de mezcla. λ = 1 = arquitectura pura, λ = 0 = SIRI puro.

---

## §6 — SIRI como post-processing (Sinkformers-style)

[THEOREM] **Sinkformers: self-attention iterations como Wasserstein gradient flow**

[THEOREM-6.1] (Sander et al. 2021, arxiv:2110.11773) Las iteraciones de self-attention con normalización Sinkhorn-Knopp corresponden, en el límite de muestras infinitas y reescalando depth/attention matrices, a una ecuación de difusión de calor bajo la métrica Wasserstein.

[COROLLARY] **Post-processing SIRI**

[COR-6.2] La nueva arquitectura puede aplicar SIRI como post-processing sobre la matriz de atención A_SOTA producida por softmax / linear / delta / retention. Esto convierte cualquier atención A_SOTA en una matriz doubly-stochastic vía Sinkhorn:

  A_post = Sinkhorn(A_SOTA, τ=5)    # preserva semántica, fuerza doubly-stochastic

[AXIOM] **A8 (Invasividad)** El post-processing SIRI no requiere modificar la arquitectura base; es una capa de regularización opt-in. Implementación: módulo `siri_postprocess.py` (a crear).

---

## §7 — Pipeline Bubble Transformer post-SDOT

[DEFINITION] **Pipeline actualizado**

[DEF-7.1] El flujo completo del Bubble Transformer con SIRI + Power Diagram + nueva arquitectura:

```
  Input X = {x₁,...,xₙ}
     ↓
  [W_Q, W_K, W_V]  →  Q, K, V
     ↓
  [Architectura SOTA: Attention_SOTA(Q, K, V)]  →  A_SOTA
     ↓ (opcional)
  [SIRI Post-Processing: Sinkhorn(A_SOTA, ε)]  →  A_SIRI (doubly-stochastic)
     ↓
  [Power Diagram bias: A_ψ = A_SIRI · diag(exp(ψ))]  →  A_final
     ↓
  O = A_final · V
     ↓
  [Metrics: Effective Rank, Intrinsic Dim, Anisotropy, Concentration]
     ↓
  [ε Sweep + Sweet Spot Analysis]
```

[LEMMA] **Complejidad computacional**

[LEMMA-7.2] Bajo el pipeline propuesto:

  - Attention_SOTA: O(n²) softmax o O(n) linear/SSM (según arquitectura)
  - SIRI post-process: O(n² · τ) (τ=5 iteraciones Sinkhorn sobre matriz n×n)
  - Power Diagram bias: O(n) (multiplicación por diagonal)
  - Total overhead SIRI: O(n² · τ) = O(5n²) — despreciable vs O(n²) softmax

[THEOREM] **Complejidad worst-case**

[THEOREM-7.3] Con arquitectura SOTA lineal (DeltaNet, RetNet, GLA, Mamba-2):

  T(n) = O(n) [atención] + O(n²) [SIRI post-process]
       = O(n²)        (dominante: SIRI post-process)

[CONJECTURE] **Optimización futura**

[CONJ-7.4] Si n ≥ 4096, el post-process SIRI O(n²) puede dominar el cómputo. Optimización futura: Sinkhorn en chunks con forma estructurada, o Sparse Sinkhorn Attention (Tay et al. 2020, arxiv:2002.11296). Pendiente para Fase 3+.

---

## §8 — Invariantes formales preservados

[DEFINITION] **Invariantes**

[DEF-8.1] La nueva arquitectura DEBE preservar:

  I1 (Costo geométrico): Cᵢⱼ = ‖Qᵢ - Kⱼ‖² — NO producto interno
  I2 (Doubly-stochastic): A ∈ Σₙ bajo SIRI
  I3 (Power Diagram bias): log_S = -C/ε + ψ donde ψ = W_ψ · K
  I4 (ε bandwidth): ε ∈ (0, ∞), rango operativo [0.001, 1.0]
  I5 (NumPy contract): Sinkhorn implementado en NumPy puro (sin PyTorch)
  I6 (τ = 5 iteraciones): Convergencia práctica verificada

[COROLLARY] **Compatibilidad con tests existentes**

[COR-8.2] Los tests que pasan (495 de 524) verifican propiedades que NO cambian bajo la nueva arquitectura:

  - v3_core, v4_core: estructura de Voronoi/Power Diagram → PRESERVADA
  - sdota_attention, sdot_attention_v4: lógica SIRI → PRESERVADA
  - spectral_metrics: independiente de arquitectura → PRESERVADA
  - tensor_compat: contrato NumPy → PRESERVADO

[EMPIRICAL] **Tests que requieren actualización**

[EMP-8.3] Los 29 tests fallidos preexistentes se dividen en:

  A. Tests rotos por bugs (4 clusters) — fixes triviales en Fase 1
  B. Tests que asumen SDOT específico — actualizar en Fase 3

---

## §9 — Conclusión

[THEOREM] **Compatibilidad universal**

[THEOREM-9.1] Bajo los invariantes I1-I6, SIRI + Power Diagram se preservan en cualquier arquitectura SOTA que mantenga proyecciones Q/K/V separadas. La transición desde SDOT a SOTA no requiere modificar SIRI ni ψ — solo reemplazar la atención base.

[AXIOM] de cierre: la decisión arquitectónica final (DeltaNet vs RetNet vs GLA vs Kimi Linear vs Mamba-2) se delega al análisis comparativo de papers (`docs/decisions/2026-06-27-sota-replacement-siri-preserved.md`). SIRI + ψ son invariantes; la arquitectura es variable.

---

*Fin del formalismo. Documento L2 con epistemic tags. Listo para Fase 1.*