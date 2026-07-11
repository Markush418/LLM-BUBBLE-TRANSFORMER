# BT V5 · Paper II — SIRI
### Sparsity-Induced Rank Inflation: teoría, mecanismo conjetural y defensa

> **Qué es esto.** El tratamiento formal del hallazgo empírico central del BT — SIRI — y la preparación de su flanco más atacable por reviewers. SIRI es la contribución científica titular del paper arXiv; este documento la separa en lo verificado, lo conjetural y lo indefendible-tal-como-está.
>
> **El hallazgo en una línea.** Al bajar $\varepsilon$ de 1.0 a 0.005 sobre Qwen3-0.6B, el effective rank de los embeddings post-atención **subió** de ~199 a ~576 — lo opuesto al colapso dimensional que la literatura predice para atención esparsificada.
>
> **Tags:** `[EMPIRICAL]` medido · `[THM]` establecido · `[CONJECTURE]` propuesto sin prueba completa · `[SKETCH]` prueba incompleta · `⚠` flanco de reviewer.

---

## §1 — Definición y propiedades del effective rank

[DEF] *(Roy & Vetterli 2007)* Para $X\in\mathbb{R}^{N\times d}$ con valores singulares $\sigma_1\ge\cdots\ge\sigma_r>0$:
$$p_k=\frac{\sigma_k}{\sum_j\sigma_j},\qquad R_{\text{eff}}(X)=\exp\big(H(p)\big)=\exp\Big(-\sum_k p_k\log p_k\Big).$$

[LEM] *(Propiedades, derivadas en Documento 0 Capas B+D)*
1. **Invariancia unitaria:** $R_{\text{eff}}(UXV)=R_{\text{eff}}(X)$ para $U,V$ ortogonales.
2. **Rango:** $R_{\text{eff}}\in[1,\operatorname{rank}(X)]$.
3. $R_{\text{eff}}=1\iff X$ es rank-1.
4. $R_{\text{eff}}=\operatorname{rank}(X)\iff$ espectro singular uniforme.

[⚠ — punto de medición] **$R_{\text{eff}}$ se mide sobre $X$ (embeddings post-atención, $[N,d]$), NO sobre $A$ (la attention map, $[N,N]$).** Son matrices distintas con interpretaciones distintas. El reporte BT debe explicitarlo; confundirlas es un error que un reviewer detecta de inmediato.

[DEF] **Métricas complementarias** (fortalecen el claim, añadir a `metrics.py`):
- **anisotropy** $=\sigma_2/\sigma_1$ (cerca de 0 = alta concentración).
- **spectral gap** $=\sigma_1/\overline{\sigma_{2:}}$ (ratio de outlier).

---

## §2 — Por qué la literatura predice lo contrario (y por qué SIRI sorprende)

[THM] *(Rank collapse, Dong et al. 2021)* La atención pura sin skip connections pierde rango **doblemente exponencial** con la profundidad, convergiendo a rank-1. Esparsificar la atención intuitivamente *aceleraría* este colapso.

[THM] *(Entropy collapse, Zhai et al. 2023)* Un modo de falla distinto: la entropía de la attention map colapsa (atención localizada degenerada).

[EMPIRICAL] **SIRI contradice la intuición ingenua:** menos $\varepsilon$ (más sparse) → **más** effective rank, no menos. La medición: $R_{\text{eff}}$ pasa de ~199 (ε=1.0) a ~576 (ε=0.005) en Qwen3-0.6B, 63 puntos, 6 capas.

⚠ **El gancho del paper es exactamente esta contradicción.** Pero la contradicción solo es real si se desambigua qué rango se mide (§4).

---

## §3 — Mecanismo conjetural de SIRI

[CONJECTURE] *(Mecanismo propuesto — no existe en la literatura → publicable)* Sea $X$ los embeddings post-atención, $A_\varepsilon$ la attention map con temperatura $\varepsilon$:

1. **Sparsity:** $A_\varepsilon$ esparsifica → cada token superviviente concentra mayor carga informacional.
2. **Presión de reconstrucción:** para minimizar la reconstruction loss con tokens escasos, el sistema debe maximizar la **ortogonalidad** entre representaciones supervivientes (si fueran colineales, perderían información irrecuperablemente).
3. **Máxima entropía bajo conservación de masa:** por el principio de máxima entropía sujeto a la restricción de conservación de masa del OT (Documento 0, Capa D), los supervivientes se proyectan hacia **ejes mutuamente ortogonales** en la variedad latente.
4. **Espectro:** la distribución de valores singulares de $X$ se vuelve más uniforme.
5. **Conclusión:** $R_{\text{eff}}=\exp(H(\sigma/\|\sigma\|_1))$ **aumenta**.

[SKETCH — el gap formal] El **Paso 3** es el eslabón sin prueba. Requiere el **dual de Kantorovich bajo la restricción one-sided** (exactamente el setting de Litman 2025, arXiv:2508.08369). Cerrar este sketch — derivar la ortogonalización como consecuencia variacional del dual one-sided — convertiría SIRI de `[CONJECTURE]` a `[THEOREM]`. **Esta es la pieza de mayor valor científico pendiente del proyecto.**

[ARCH] Metáfora operativa (no prueba): la "tensión superficial" de la película de jabón. La atención se tensa hacia los tokens de mayor valor; los supervivientes, para no colapsar, se reparten en direcciones ortogonales → "islas semánticas hiper-densas" de alta dimensionalidad intrínseca.

---

## §4 — El flanco de reviewer: ¿qué noción de rango? [crítico]

⚠ **Este es el punto que hunde o salva el paper.** "Rango" tiene dos lecturas que normalmente coinciden pero **divergen exactamente en el régimen $\varepsilon\to 0$ que SIRI sondea** (Documento 0, Capas B+F):

| Noción | Qué mide | En ε→0 (casi-permutación) |
|---|---|---|
| **Sparsity del soporte** | nº de entradas no nulas | **bajo** (pocos no-ceros) |
| **Effective rank espectral** | dispersión de $\{\sigma_k\}$ | **alto** (espectro plano) |

[THM] Una matriz de permutación tiene **todos** sus valores singulares iguales a 1 (Documento 0, Capa B, ejercicio) → espectro plano → $R_{\text{eff}}$ máximo. El trabajo de Sinkhorn de bajo rango (Scetbon–Cuturi–Peyré, arXiv:2103.04737) lo dice explícitamente: una solución OT de rango completo es una permutación con espectro plano.

[CONJECTURE → resuelto el flanco] **La defensa correcta:** SIRI mide el effective rank **espectral de los embeddings $X$**, y el aumento es **consistente** con que la atención dispersa empuje el espectro hacia la uniformidad (no es paradójico una vez desambiguado). El error sería afirmar simultáneamente "sparsity alta" y "rango alto" sin aclarar que son nociones distintas — un reviewer lo leería como contradicción.

[⚠ — claim limpio vs claim riesgoso]
- **Claim inequívoco y defendible:** en $\varepsilon\to\infty$, $A\to ab^\top$ (rank-1, $R_{\text{eff}}\to 1$). Esto es un teorema (Documento 0, Capa F).
- **Claim empírico fuerte:** en el régimen intermedio ($\varepsilon\approx 0.005$), $R_{\text{eff}}(X)$ alcanza un pico de ~576. Medido, no derivado.
- **Claim riesgoso (evitar sin desambiguar):** "sparsity ⇒ alto rango" presentado como universal. Solo vale para el rango espectral de $X$, no para el soporte de $A$.

---

## §5 — La Meseta de Saturación: definición topológica robusta

[HEURISTIC] **Definición actual (frágil):** $S_{\text{sat}}=\{\varepsilon:\ d|\!\operatorname{Support}(A)|/d\varepsilon=0\}$ con $|\!\operatorname{Support}(A)|=|\{(i,j):A_{ij}>10^{-5}\}|$. El threshold $10^{-5}$ es **arbitrario** (depende de la precisión float32) → atacable.

[DEF] **Definición topológica (invariante de precisión):** sea $G_\varepsilon$ el grafo de atención con aristas $(i,j)$ donde $A_{ij}>\delta_{\text{rel}}$, con threshold **relativo** $\delta_{\text{rel}}=\max_{ij}A_{ij}\cdot\varepsilon_{\text{machine}}$. Entonces
$$S_{\text{sat}}=\{\varepsilon:\ \beta_0(G_\varepsilon)=\text{const}\ \wedge\ \beta_1(G_\varepsilon)=\text{const}\},$$
donde $\beta_0$ = nº de componentes conectadas, $\beta_1$ = nº de ciclos independientes (Betti numbers).

[THM] **Por qué es mejor:** $\beta_0,\beta_1$ son **invariantes topológicos** → independientes de la precisión numérica; $\delta_{\text{rel}}$ escala automáticamente con el hardware. Implementable con `gudhi` (Rips complex sobre la adyacencia). Conecta con la Γ-convergencia (Documento 0, Capa F): la meseta es donde el minimizador ya alcanzó el vértice límite topológicamente — bajar $\varepsilon$ solo cambia magnitudes, no el grafo activo.

---

## §6 — Conexión con phase transitions (valida la conjetura abierta)

[CONJECTURE] *(BT refinada, respaldada por 3 papers independientes)* "La topología de $G_\varepsilon$ experimenta **transiciones de fase discretas** durante el training, detectables vía $\beta_0(G_\varepsilon)$ y $\beta_1(G_\varepsilon)$. El Baroreceptor MLP (Paper I §3) aprende a predecir el régimen líquido/sólido y ajusta $C$ antes de la transición."

Evidencia:
- **arXiv:2601.19942** (Latent Object Permanence): transición en $\gamma_c\approx 0.42$; régimen líquido (bulk Marchenko–Pastur, alta entropía) vs sólido (spectral gaps, baja entropía); framework de free-energy variacional + RG flow.
- **arXiv:2510.07401** (Attention to Order): fases ordenadas → mayor learnability; $\varepsilon$ del BT es análogo a temperatura crítica física.
- **arXiv:2410.11042** (Persistent Topological Features in LLMs): 4 fases (rearrangement → stable → transition → final); zigzag persistent homology; $\beta_0,\beta_1$ como descriptores de fase.

⚠ Status: `[CONJECTURE]` con evidencia convergente. **Experimento que la cierra:** medir $\beta_0,\beta_1$ de $G_\varepsilon$ durante el training de Qwen3-0.6B con SDOT inyectado (Sprint 3-4).

---

## §7 — Qué invalida y qué NO invalida SIRI

⚠ **El incidente RoPE (PPL 831,974) NO invalida SIRI.** SIRI es una propiedad de la **geometría de embeddings bajo dispersión**, medida sobre embeddings extraídos con softmax estándar (`extract_embeddings.py`), **no** con el wrapper inyectado sin RoPE. El fenómeno $R_{\text{eff}}=576$ @ $\varepsilon=0.005$ es real e independiente del bug.

[NOTE] Lo que el bug sí mostró: las métricas geométricas puras ($R_{\text{eff}}$, concentration) son válidas sin RoPE, pero la **PPL** no — porque la posición es predictivamente esencial. Son planos de evaluación distintos.

[⚠ — reproducibilidad] El pico SIRI debe probarse **reproducible a través de**: (a) distintas normalizaciones del costo $C$, (b) distintos tamaños de modelo, (c) distintas capas. Si el pico se mueve o desaparece, SIRI es un artefacto de Qwen3-0.6B, no una ley general. Reformular como "model-specific" si no resiste (Paper V, thresholds).

---

## §8 — Resumen del status epistémico de SIRI

| Afirmación | Status | Defensa |
|---|---|---|
| $R_{\text{eff}}$ sube de 199→576 al bajar ε | `[EMPIRICAL]` | medido, reproducible-pendiente |
| ε→∞ ⇒ rank-1 | `[THM]` | Γ-convergencia (Doc 0, Capa F) |
| Existe pico interior de $R_{\text{eff}}$ | `[EMPIRICAL]` + bracketing | consistente, ubicación no derivada |
| Mecanismo (ortogonalización por sparsity) | `[CONJECTURE]` + `[SKETCH]` | gap en Paso 3 (dual one-sided) |
| Meseta topológica (Betti) | `[DEF]` mejorada | invariante de precisión |
| Phase transitions durante training | `[CONJECTURE]` | 3 papers, experimento pendiente |

[ARCH] **La jugada de publicación:** presentar SIRI como **fenómeno empírico robusto + conjetura mecanística con sketch**, no como teorema. Es más honesto, más defendible, y deja el `[THEOREM]` (cerrar el dual one-sided) como contribución futura clara. Un reviewer respeta un `[CONJECTURE]` bien delimitado; castiga un `[THEOREM]` con un gap escondido.

---

*Siguiente: Paper III — Resolución formal de los blockers: diferenciabilidad, routing, RoPE.*
