# BT V5 · Documento 0 — Bases Primarias
### El lecho de roca matemático bajo los pilares de estudio

> **Propósito.** El documento de estudio anterior (`bases_matematicas_estudio`) cubre los **pilares operativos** [1]–[7]: atención, SVD, entropía, OT/Sinkhorn, geometría, diferenciabilidad, complejidad. Este documento va **una capa más abajo**: los axiomas y teoremas de existencia sobre los que esos pilares se apoyan. Si los pilares son "cómo se usa", esto es "por qué tienen derecho a existir".
>
> **A quién sirve.** A quien quiera *defender* el BT ante un reviewer, no solo usarlo. Cada teorema de existencia aquí es una pregunta que un revisor puede hacer ("¿el plan de transporte óptimo *existe*?", "¿la entropía es *la única* medida razonable?", "¿el límite ε→0 *converge* a algo?").
>
> **Tags:** `[AXIOM]` · `[DEF]` · `[THM]` · `[LEM]` · `[PROP]` · `[NOTE]` · `⚠` flanco atacable · `[BT]` conexión directa.

---

## Capa A — Espacios vectoriales con producto interno

Todo (embeddings, queries, keys, centroides) vive en $\mathbb{R}^d$ con su estructura euclidiana. No es un detalle decorativo: la geometría de Voronoi, el costo de transporte $\|Q_i-K_j\|^2$ y la ortogonalidad de SIRI **dependen** de que haya un producto interno.

[AXIOM] Un **espacio con producto interno** real es un $\mathbb{R}$-espacio vectorial $V$ con una forma $\langle\cdot,\cdot\rangle: V\times V\to\mathbb{R}$ bilineal, simétrica y definida positiva ($\langle x,x\rangle>0$ para $x\neq 0$).

[DEF] La **norma** inducida es $\|x\|=\sqrt{\langle x,x\rangle}$, y la **distancia** $d(x,y)=\|x-y\|$.

[THM] *(Cauchy–Schwarz)* $|\langle x,y\rangle|\le\|x\|\,\|y\|$, con igualdad sii $x,y$ son colineales.

[BT] Consecuencias que el BT usa sin enunciarlas:
- La **matriz de costo** $C_{ij}=\|Q_i-K_j\|^2$ está bien definida y es no negativa → es un costo de OT legítimo.
- La **ortogonalidad** ($\langle x,y\rangle=0$) que invoca el mecanismo de SIRI ("los tokens supervivientes se proyectan en ejes mutuamente ortogonales") solo tiene sentido porque hay producto interno.
- La asignación Voronoi $\arg\min_i\|x-c_i\|$ requiere que $d$ sea una métrica genuina (desigualdad triangular incluida).

⚠ Cuando el BT mueve los centroides al **Disco de Poincaré** (geometría hiperbólica, ver Documento 1 §6), el producto interno euclidiano deja de valer y se reemplaza por la **métrica Riemanniana** $g_x$ del manifold. Las propiedades de arriba se reformulan localmente (en el espacio tangente $T_xV$). Esto es exactamente lo que `geoopt` administra.

---

## Capa B — El teorema espectral y la existencia de la SVD

El pilar [2] (SVD, effective rank) descansa en que **toda matriz tiene SVD**. Esto no es obvio; se deriva del teorema espectral para operadores simétricos.

[THM] *(Teorema espectral, caso simétrico real)* Si $M\in\mathbb{R}^{n\times n}$ es simétrica, existe una base ortonormal de autovectores: $M=Q\Lambda Q^\top$ con $Q$ ortogonal y $\Lambda$ diagonal real.

[THM] *(Existencia de la SVD)* Para **cualquier** $M\in\mathbb{R}^{m\times n}$, aplicar el teorema espectral a $M^\top M$ (que es simétrica y semidefinida positiva) produce $M=U\Sigma V^\top$ con $\sigma_k=\sqrt{\lambda_k(M^\top M)}\ge 0$.

[PROP] Los valores singulares son **únicos** (el orden $\sigma_1\ge\cdots\ge\sigma_r$ los fija); $U,V$ no lo son si hay $\sigma$ repetidos.

[BT] Por qué importa para SIRI:
- El effective rank $R_{\text{eff}}=\exp(H(\sigma/\|\sigma\|_1))$ está **bien definido y es único** porque los $\sigma_k$ lo están.
- La invariancia unitaria $R_{\text{eff}}(UMV)=R_{\text{eff}}(M)$ (que el audit del BT lista como propiedad) es un **corolario directo** de que $U,V$ ortogonales no alteran el espectro singular.
- $R_{\text{eff}}\in[1,\operatorname{rank}(M)]$ con los dos extremos caracterizados (rank-1 ↔ $R_{\text{eff}}=1$; espectro plano ↔ $R_{\text{eff}}=\operatorname{rank}$) sale de que $H$ alcanza su mínimo en distribuciones degeneradas y su máximo en la uniforme (Capa D).

---

## Capa C — El símplex de probabilidad y la estructura de las distribuciones

Atención, planes de transporte y asignaciones de burbuja son todos **objetos de probabilidad**. El espacio donde viven tiene estructura geométrica propia.

[DEF] El **símplex de probabilidad** $n$-dimensional es
$$\Delta^n=\Big\{p\in\mathbb{R}^n : p_i\ge 0,\ \textstyle\sum_i p_i=1\Big\}.$$
Es un conjunto **convexo y compacto** (cerrado y acotado).

[DEF] El **polytope de transporte** $U(a,b)=\{P\ge 0: P\mathbf1=a,\ P^\top\mathbf1=b\}$ es la intersección de $\Delta$-tipo con restricciones lineales → también **convexo y compacto y no vacío** (contiene a $ab^\top$).

[THM] *(Birkhoff–von Neumann)* El caso $a=b=\tfrac1n\mathbf1$ da el **polytope de Birkhoff** $\mathcal B_n$ de matrices doblemente estocásticas; sus **vértices son exactamente las matrices de permutación**.

[BT] Esto fundamenta tres cosas:
1. Una matriz de atención (fila-estocástica) es un punto en un producto de símplices $\Delta^n\times\cdots$; el BT la empuja hacia $\mathcal B_n$ (doblemente estocástica).
2. La afirmación "el límite ε→0 del OT es un **vértice** del polytope" (una casi-permutación, base de la Meseta de Saturación) **es** Birkhoff–von Neumann: los óptimos de un programa lineal sobre un politopo se alcanzan en vértices.
3. La compacidad garantiza que los mínimos de OT **existen** (Capa E).

⚠ El BT V3/V4 usa atención *one-sided* (solo fila-estocástica, no doblemente). Esto lo saca técnicamente del polytope de Birkhoff y lo pone en el setting de Litman (2025, arXiv:2508.08369). La distinción one-sided vs doubly-stochastic es central y se trata en el Documento 2.

---

## Capa D — Fundamento axiomático de la entropía

El pilar [3] usa $H(p)=-\sum p_k\log p_k$ como si fuera *la* medida de dispersión. ¿Por qué esa fórmula y no otra? Porque es la **única** (salvo escala) que satisface tres axiomas razonables.

[THM] *(Teorema de unicidad de Shannon, 1948)* Una función $H(p_1,\dots,p_n)$ que satisface
1. **continuidad** en los $p_i$,
2. **monotonía**: para la uniforme, $H(\tfrac1n,\dots,\tfrac1n)$ crece con $n$,
3. **aditividad/agrupamiento**: la entropía de una elección compuesta es la suma ponderada de las parciales,

es necesariamente $H(p)=-\kappa\sum_i p_i\log p_i$ para alguna constante $\kappa>0$.

[LEM] *(Cotas)* $0\le H(p)\le\log n$. Mínimo en distribuciones degeneradas ($H=0$); máximo en la uniforme ($H=\log n$). *Prueba del máximo:* concavidad de $H$ + Lagrange con la restricción $\sum p_i=1$ (ejercicio estándar).

[BT] Implicaciones:
- El uso de $H$ en el **effective rank** (Capa B) no es arbitrario: es la única medida de "cuántos modos importan" consistente con los axiomas.
- El uso de $H$ como **regularizador entrópico** del OT (Capa E) hereda la concavidad estricta → garantiza unicidad de la solución de Sinkhorn.
- La **maximización de entropía bajo restricciones de conservación de masa** es el principio que el mecanismo de SIRI invoca en su paso 3 (Documento 2). Ese principio es legítimo precisamente por la unicidad de Shannon.

---

## Capa E — Análisis convexo, dualidad y existencia del transporte óptimo

El pilar [4] (OT/Sinkhorn) es donde más peso soporta la estructura. Tres resultados lo sostienen.

### E.1 Existencia del plan óptimo

[THM] *(Existencia, Kantorovich)* Como $U(a,b)$ es **compacto no vacío** (Capa C) y el funcional $P\mapsto\langle P,C\rangle$ es **continuo**, el mínimo $\min_{P\in U(a,b)}\langle P,C\rangle$ **se alcanza** (Weierstrass). El plan óptimo existe.

[THM] *(Existencia + unicidad, caso entrópico)* El funcional regularizado $\langle P,C\rangle-\varepsilon H(P)$ es **estrictamente convexo** (la $-\varepsilon H$ lo es, por la concavidad estricta de $H$, Capa D) sobre un convexo compacto → el mínimo existe y es **único**. Este es $P^\varepsilon$.

### E.2 Dualidad de Lagrange y condiciones KKT

[DEF] El **Lagrangiano** del OT entrópico, introduciendo multiplicadores $f\in\mathbb{R}^n$, $g\in\mathbb{R}^m$ para las dos restricciones marginales:
$$\mathcal L(P,f,g)=\langle P,C\rangle-\varepsilon H(P)-\langle f,P\mathbf1-a\rangle-\langle g,P^\top\mathbf1-b\rangle.$$

[THM] *(Forma de la solución vía KKT)* Anulando $\partial\mathcal L/\partial P_{ij}=0$ se obtiene
$$P_{ij}^\varepsilon=\underbrace{e^{f_i/\varepsilon}}_{u_i}\;\underbrace{e^{-C_{ij}/\varepsilon}}_{K_{ij}}\;\underbrace{e^{g_j/\varepsilon}}_{v_j},$$
es decir $P^\varepsilon=\operatorname{diag}(u)\,K\,\operatorname{diag}(v)$. **Esta es la forma escalada que el algoritmo de Sinkhorn busca** — no un truco algorítmico sino la condición de optimalidad de primer orden.

[NOTE] Las condiciones KKT (estacionariedad, factibilidad primal/dual, holgura complementaria) son el marco general de la optimización con restricciones; aquí solo se usa la estacionariedad porque el problema es convexo con restricciones de igualdad.

### E.3 Convergencia de Sinkhorn

[THM] *(Sinkhorn–Knopp / Franklin–Lorenz)* Para $K$ de entradas estrictamente positivas, la iteración alterna $u\leftarrow a\oslash(Kv)$, $v\leftarrow b\oslash(K^\top u)$ es una **contracción** en la **métrica de Hilbert proyectiva**; converge linealmente al único par $(u,v)$. *(La métrica de Hilbert convierte el escalado positivo en una contracción estricta; el factor de contracción se controla por la razón de condición de $K$.)*

[BT] El "Impuesto de Iteración Secuencial" (IIS / Impuesto de Jersey) del BT es la **constante** de esta convergencia traducida a hardware: cada iteración es un par de productos matriz-vector con dependencia secuencial en HBM. El BT no acelera la convergencia de Sinkhorn — la **elimina**, reemplazando el escalado iterativo por una partición geométrica directa (el límite ε→0 es una casi-permutación, Capa C, computable como Power Diagram en un solo paso). Ese es el argumento de diseño completo del BT en una frase.

---

## Capa F — Convergencia de los límites en ε (el rigor bajo SIRI y la Meseta)

El documento de estudio afirma cosas sobre "$\varepsilon\to\infty$" y "$\varepsilon\to 0$". Esos límites necesitan un marco que garantice que **convergen** y **a qué**.

[THM] *(Límite ε→∞)* $K=e^{-C/\varepsilon}\to\mathbf1\mathbf1^\top$ entrada a entrada (continuidad de la exponencial), de donde el escalado fuerza $P^\varepsilon\to ab^\top$. Convergencia **puntual** directa. → producto de marginales, rank-1.

[THM] *(Límite ε→0, Γ-convergencia)* El funcional entrópico **Γ-converge** al funcional del OT no regularizado cuando $\varepsilon\to 0$ (Cominetti–San Martín 1994; Peyré–Cuturi 2019). La Γ-convergencia es la noción correcta: garantiza que los **minimizadores** $P^\varepsilon$ convergen a un minimizador del problema límite (un vértice del polytope, Capa C), no solo que los valores óptimos convergen.

[BT] Esto le da piso formal a dos afirmaciones del BT:
- La **Meseta de Saturación** $S_{\text{sat}}=\{\varepsilon:\ d\,|\!\operatorname{Support}(A)|/d\varepsilon=0\}$ es el régimen donde el minimizador ya alcanzó (topológicamente) el vértice límite: bajar más $\varepsilon$ no cambia *qué* aristas están activas, solo sus magnitudes. La versión topológica vía Betti numbers ($\beta_0,\beta_1$ constantes) que propone el audit es una formalización **invariante de precisión** de exactamente este hecho.
- El **pico de SIRI** en $\varepsilon\approx 0.005$ vive en el régimen intermedio de la Γ-convergencia: ni el colapso rank-1 ($\varepsilon\to\infty$) ni el vértice disperso ($\varepsilon\to 0$). Que exista un interior con $R_{\text{eff}}$ máximo es consistente con el bracketing, pero —⚠— la **ubicación** del pico no se deriva de ningún teorema; es empírica y depende de la normalización de $C$. Esto se trata en detalle en el Documento 2.

---

## Síntesis — el grafo de fundamentación completo

```
Capa A (producto interno)
   ├─→ costo OT ‖Q-K‖²            ──┐
   ├─→ distancia Voronoi            │
   └─→ ortogonalidad (SIRI)         │
Capa B (espectral → SVD)            │
   └─→ effective rank único        ─┤
Capa C (símplex / Birkhoff)         │
   ├─→ existencia de planes         ├─→ Pilar [4] OT/Sinkhorn
   └─→ vértices = permutaciones    ─┤      (NÚCLEO del BT)
Capa D (axiomas de Shannon)         │
   ├─→ H es la única medida         │
   └─→ concavidad → unicidad       ─┤
Capa E (convexidad/KKT/contracción) │
   ├─→ existe y es único P^ε        │
   ├─→ forma diag(u)K diag(v)       │
   └─→ Sinkhorn converge           ─┘
Capa F (Γ-convergencia en ε)
   ├─→ ε→∞ ⇒ rank-1 (riguroso)
   └─→ ε→0 ⇒ vértice (Meseta, SIRI bracketing)
```

[BT] **Tesis de fundamentación del BT en una línea:** el BT es legítimo porque (A) hay una geometría que define costos, (C+E) el transporte óptimo entrópico existe y es único con forma escalada, (F) su límite $\varepsilon\to 0$ converge a una partición geométrica computable en un paso (Power Diagram), y (B+D) el fenómeno emergente (SIRI) se mide con la única noción de rango consistente con los axiomas de la información. Cada eslabón de esa cadena es un teorema, no una conjetura — **salvo** la ubicación del pico SIRI y el mecanismo causal detrás de él, que son las contribuciones empíricas/conjeturales propias del proyecto.

---

*Siguiente: Documento 1 — Arquitectura real del BT V5 (forward pass formal, integrando todos los hallazgos del chat).*
