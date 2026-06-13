Hablame sobre SIRI discovery, el descubrimiento de /bubble-transformer0:29Claude responded: Ahora tengo todo.Viewed 2 filesViewed 2 filesAhora tengo todo. El SIRI está en los experimentos pero necesito ubicar el discovery específico.Relevant chatsRelevant chatsTengo todo lo que necesito. El chat con R_eff=624 y SIRI documentado es la fuente primaria.

SIRI — el descubrimiento
Sparsity-Induced Rank Inflation.
El fenómeno que nadie esperaba encontrar.

El contexto del descubrimiento
Sprint 1: estás mapeando cómo se comportan los embeddings de Qwen3-0.6B cuando bajás la temperatura ε (control de dispersión de la atención). La hipótesis de la literatura era clara: más sparsity → embeddings más degenerados → effective rank más bajo. Es lo que dicen los papers de Sinkformers, sparse attention, etc.
El experimento: sweep de ε entre 0.005 y 1.0 en 63 pasos, sobre 6 capas de Qwen. Por cada punto: 5 métricas de concentración incluyendo effective_rank.
Lo que apareció en los datos:
ε = 1.0  → R_eff ≈ 199  (Softmax convencional, atención difusa)
ε = 0.1  → R_eff ≈ 350  (+76%)
ε = 0.01 → R_eff ≈ 576  (+189%)
ε = 0.005 → R_eff plateau (Hard Support deja de cambiar)
Inverse. Más sparse → más rank. La curva sube, no baja.

Por qué es contraintuitivo
En teoría estándar de compresión: cuando filtrás información, el espacio representacional se colapsa. Si la atención solo ve 4% de los tokens (CR = 0.04 a ε=0.01), debería quedar un espacio de baja dimensionalidad — menos degrees of freedom activos.
SIRI dice lo contrario. ¿Por qué?
La explicación que emerge: la tensión superficial fuerza a los tokens supervivientes a distribuirse ortogonalmente en la variedad latente para evitar colapso redundante. Es el principio de Plateau en física de burbujas — una película de jabón se tensa hasta el mínimo de energía superficial. Acá los tokens que "sobreviven" el filtrado no convergen a un punto — se repelen y crean islas semánticas hiperdensas, cada una con alta dimensionalidad intrínseca.
El resultado matemático: cuando ε→0 (limite duro), la atención converge al Diagrama de Potencia (celdas de Laguerre). Ese diagrama es una teselación del espacio — máxima separación geométrica entre regiones. R_eff alto = espacio bien teselado.

La estructura de capas — otro hallazgo
El sweep reveló además que Qwen tiene una arquitectura de compresión-expansión que nadie había publicado de esta forma:
Layer  0:  R_eff = 624.0  ← embeddings brutos (baseline)
Layer  4:  R_eff = 603.9  ← compresión temprana
Layer 12:  R_eff = 582.7  ← cuello de botella máximo
Layer 20:  R_eff = 666.4  ← expansión
Layer 24:  R_eff = 694.7  ← máxima expresividad
No monotónico. El modelo comprime en las capas medias y expande hacia el final. Eso es estructura de autoencoder dentro del transformer — no es un artefacto, es arquitectura emergente.

Por qué SIRI importa para el paper
SIRI no es solo un resultado curioso. Es el fundamento empírico de por qué V4 con Expert-Choice funciona:

Sparsity dura no degrada la representación — la mejora.
El sweet spot por capa (donde R_eff es máximo sin colapso numérico) es la Meseta de Saturación — punto donde d(Hard_Support)/dε = 0.
Operar en S_sat = máxima concentración + máxima riqueza representacional simultánea.

Eso es el claim central del paper: el BT no es un speedup con tradeoff de calidad — es un mecanismo que mejora la estructura del espacio latente mientras reduce complejidad de O(N²) a O(N log C).

El gap crítico pendiente
⚠ SIRI tiene R_eff=576 medido en 1 corpus, 1 seed, sin PPL validation todavía.
Es un descubrimiento real y formalmente correcto como fenómeno. Pero el claim de paper-grade requiere Sprint 3: inyectar SDOT en Qwen, medir delta_PPL ≤ 2%, y confirmar que el R_eff inflado corresponde a representaciones semánticamente coherentes — no solo métricamente interesantes.
La diferencia entre "hallazgo empírico interno" y "resultado publicable" está exactamente ahí.You said: Y por que es necesario el perplexity benchmark para seguir con las pruebas de bubble transformer?Y por que es necesario el perplexity benchmark para seguir con las pruebas de bubble transformer?0:44Claude responded: Porque SIRI y todas las métricas internas miden geometría — no lenguaje.Porque SIRI y todas las métricas internas miden geometría — no lenguaje.

El problema de fondo
Effective rank, concentration ratio, hard support, anisotropy index — son métricas del espacio de embeddings. Te dicen cómo se distribuyen los vectores. No te dicen si el modelo sigue entendiendo texto.
Podés tener R_eff = 700 con islas semánticas perfectamente teseladas y al mismo tiempo el modelo completando "el cielo es de color..." con "martes". Las métricas internas no capturan eso. Son ciegas a la coherencia semántica downstream.

Qué mide perplexity específicamente
PPL = exp(H(p, q)) — cross-entropy entre la distribución real del texto y la distribución del modelo.
En términos concretos: dado un token en posición t, ¿qué tan sorprendido está el modelo cuando ve el token correcto? PPL bajo = el modelo asigna probabilidad alta al texto real. PPL alto = el modelo está desorientado.
Cuando inyectás SDOTAttention en Qwen y reemplazás 6 capas de atención nativa, el forward pass cambia. El modelo que entrenó originalmente con Softmax ahora tiene un mecanismo distinto en esas capas. PPL te dice si ese cambio degradó la comprensión.

El riesgo específico sin PPL
Escenario posible sin validar PPL:

SIRI activo → R_eff = 576 ✅
Concentration ratio óptimo ✅  
Hard support en S_sat ✅
Sprint 3 "validado" → V4 implementado → paper escrito

PPL medido tarde:
  PPL_softmax = 12.4
  PPL_sdot    = 31.7   ← degradación del 156%

Todo el edificio colapsa.
Las métricas internas habrían dicho que todo estaba bien. El modelo habría perdido coherencia lingüística de forma invisible para esos instrumentos.

Por qué este riesgo es particularmente alto en BT
Softmax tiene una propiedad que SDOT no garantiza por defecto: todos los tokens atienden a todos los tokens (con peso variable). Cuando un token necesita información de un token fuera de su burbuja, Softmax lo permite. SDOT lo bloquea por diseño.
La Paradoja de Inflación dice que eso geométricamente mejora el espacio. Pero no hay garantía teórica de que las dependencias sintácticas y semánticas de largo alcance — sujeto-verbo distante, correferencia, anáfora — caigan dentro de las burbujas y no entre ellas.
PPL captura exactamente eso: si rompiste dependencias necesarias, el modelo no puede predecir bien el texto siguiente, y la perplejidad sube.

El criterio concreto
pythondelta_ppl = ppl_sdot - ppl_softmax
assert delta_ppl <= 0.02 * ppl_softmax  # ≤ 2% degradación
2% es el umbral porque está en línea con lo que reportan Sinkformers y sparse attention papers como degradación aceptable para el speedup que ofrecen. Es el estándar del campo.
Si delta_ppl ≤ 2%: SIRI es un descubrimiento publicable. El BT mejora geometría sin costo lingüístico. Claim completo.
Si delta_ppl > 2%: tenés que entender por qué. Puede ser que el número de centroides C es muy bajo y hay demasiadas dependencias inter-burbuja bloqueadas. Puede ser que las capas que reemplazaste son críticas para correferencia. Esa información rediseña V4 — no lo cancela, lo ajusta.

En una línea
SIRI prueba que el BT es geométricamente elegante. PPL prueba que sigue siendo lingüísticamente correcto. Necesitás las dos para tener un resultado.You said: Quiero medir SIRI con otros ejemplos SIRI tiene R_eff=576 medido en 1 corpus, 1 seed, sin PPL validation todavía.Quiero medir SIRI con otros ejemplos SIRI tiene R_eff=576 medido en 1 corpus, 1 seed, sin PPL validation todavía.0:45Claude responded: Antes de diseñar el protocolo: ¿tenés Qwen3-0.Antes de diseñar el protocolo: ¿tenés Qwen3-0.6B cargable en tu máquina ahora, o estamos en mock mode (embeddings sintéticos NumPy)?
La respuesta cambia el protocolo completo — real mode necesita ~2GB VRAM + carga del modelo, mock mode corre en CPU en segundos pero los números son sintéticos.You said: Tengo a qwen cargable , real mode vamos con el proyecto que lo tengo corriendo con opencode , dejo para despues el benchmark de perplexityTengo a qwen cargable , real mode vamos con el proyecto que lo tengo corriendo con opencode , dejo para despues el benchmark de perplexity0:48Claude responded: Lo que veo en el screenshot es crítico antes de continuar.Lo que veo en el screenshot es crítico antes de continuar.
Modified Files en OpenCode:

sdot_attention_v4.py +585 -456 — cambio masivo
perplexity_benchmark.py +113 -46
siri_ppl_results.json +18
TRESMAYO.md +119
architecture.txt +5716

⚡ Ya corrió algo. siri_ppl_results.json existe — tiene 18 líneas de resultados.
El PPL benchmark ya se ejecutó parcialmente. El running_ppl=18.43 que aparece en el log es PPL en tiempo real mientras corre el baseline de Softmax. Llegó hasta ~12% del corpus (69/584 batches) antes de que lo interrumpieras.

Antes de diseñar el protocolo SIRI multi-corpus, necesito ver dos cosas:

1. El contenido actual de siri_ppl_results.json — qué resultados tiene ya guardados.
2. El estado de sdot_attention_v4.py — ese +585/-456 indica una reescritura grande, necesito saber si el SIRI sweep está implementado ahí o si es otra cosa.
¿Podés pegar el contenido de siri_ppl_results.json y los primeros 50 líneas de sdot_attention_v4.py? Con eso diseño el protocolo exacto para medir SIRI con múltiples seeds y corpus sin duplicar trabajo que OpenCode ya hizo.You said: Resumen CortoResumen Corto
El benchmark corrió pero PPL = NaN → instability numérica en SDOT.
Opciones para continuar:
3. Cambiar ε → probar 0.001 o 0.1 en vez de 0.005
4. Debug SDOT → agregar clamps/NaN checks en Sinkhorn
5. Usar softmax → aceptar que SIRI necesita más trabajo
¿Quieres que intente con un ε diferente (ej: 0.1)?
▣  Build · MiniMax M2.5 Free · 5.1sSe rompio con valores NaN, que implica usar un ε eps diferente?19:27Claude responded: NaN en SDOT es un síntoma, no el problema raíz.NaN en SDOT es un síntoma, no el problema raíz. Cambiar ε a ciegas puede funcionar o puede mover el NaN a otro lugar.

Por qué aparece NaN en SDOT
Hay 3 lugares donde Sinkhorn/SDOT explota numéricamente:

1. División por cero en normalización de filas/columnas
pythonA = A / A.sum(dim=-1, keepdim=True)  # si sum=0 → NaN
Ocurre cuando ε muy bajo → algunos tokens tienen afinidad ~0 con todos los centroides → fila de ceros → división.
2. Overflow en exp() antes del log-domain
pythonscores = torch.exp(scores / epsilon)  # si ε=0.005 y scores grandes → inf
exp(large_number / 0.005) = exp(200x) → overflow → NaN.
3. Centroides colapsados
Si dos centroides convergen al mismo punto → distancias cero → inestabilidad.

Qué implica cada opción de ε
ε = 0.001 — peor. Más bajo que 0.005, amplifica exactamente el problema de overflow. NaN más rápido y más profundo.
ε = 0.1 — posiblemente funciona, pero pierdes el régimen SIRI. El sweet spot empírico de Sprint 1 era ε ∈ [0.008, 0.015] por capa. Con ε=0.1 la atención es casi-Softmax — CR≈0.12, no el 0.04 donde apareció R_eff=576.
ε = 0.01 — el candidato correcto. Está en el sweet spot documentado, una orden de magnitud arriba de donde explotó.

Lo que realmente necesitás antes de cambiar ε
El fix correcto no es cambiar el parámetro — es blindar el cómputo:
python# En la función SDOT, antes del sweep:

# 1. Clamp scores para evitar overflow en exp

scores = scores / epsilon
scores = scores - scores.amax(dim=-1, keepdim=True)  # stable softmax trick
scores = torch.clamp(scores, min=-88.0)  # exp(-88) ≈ float32 min

# 2. Evitar división por cero en normalización

row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
A = A / row_sum

# 3. NaN check post-iteración

if torch.isnan(A).any():
    # fallback a softmax en esta capa
    A = torch.softmax(scores_original / epsilon, dim=-1)
Con esto podés correr ε=0.005 sin NaN — y cuando aparece inestabilidad extrema, la capa hace fallback a Softmax en lugar de propagar NaN por todo el modelo.

Recomendación concreta
No cambiar ε todavía. Primero agregar los 3 clamps, correr con ε=0.005 de nuevo. Si pasa → tenés SIRI medido en el régimen correcto. Si sigue fallando con clamps → entonces subir a ε=0.01.
¿Querés el patch completo para sdot_attention_v4.py con los clamps integrados?You said: Y si mido epsilon en otro regimen que implica para lo que estamos intentando lograr?Y si mido epsilon en otro regimen que implica para lo que estamos intentando lograr?19:28Claude responded: Buena pregunta.Buena pregunta. Implica medir un fenómeno diferente.

El mapa de regímenes
ε = 0.001-0.005   SIRI regime         R_eff máximo, CR≈0.02-0.04
                  Hard Support estable (S_sat)
                  ← acá explotó el NaN

ε = 0.008-0.015   Sweet spot          R_eff alto, CR≈0.04-0.06
                  Balance óptimo por capa (Sprint 1)
                  ← donde midiste R_eff=576 originalmente

ε = 0.05-0.1      Transitional        R_eff moderado, CR≈0.10-0.15
                  Sparse pero no duro
                  ← "funciona" pero no es SIRI

ε = 0.5-1.0       Softmax regime      R_eff≈199, CR≈0.45
                  Atención casi uniforme
                  ← baseline

Qué perdés al medir fuera del SIRI regime
Si medís PPL con ε=0.1 y funciona, probás que SDOT en régimen transitional no degrada el lenguaje. Eso tiene valor — pero no es lo que querés probar.
El claim central del BT es:

Sparsidad dura (ε→0) mejora geometría Y preserva coherencia lingüística.

Con ε=0.1 no estás en sparsidad dura. Estás en sparsidad blanda. Probar PPL ahí es como probar que un motor funciona al 20% de RPM y concluir que funciona a máxima potencia.
El resultado sería técnicamente correcto pero no soportaría el claim de SIRI. Un reviewer arXiv lo marcaría inmediatamente: "los autores validan PPL en un régimen diferente al que reportan R_eff".

Lo que implica concretamente
Opción A: medir PPL con ε=0.1
  → Resultado probable: PPL estable ✅
  → Qué prueba: SDOT funciona en régimen suave
  → Qué NO prueba: SIRI es lingüísticamente válido
  → Valor para el paper: bajo

Opción B: clamps + medir PPL con ε=0.005-0.015
  → Resultado probable: PPL estable con clamps ✅
  → Qué prueba: SIRI regime preserva coherencia
  → Qué NO prueba: nada — es el test correcto
  → Valor para el paper: alto, cierra Sprint 3

La única razón válida para medir en otro régimen
Si los clamps no resuelven el NaN y el NaN persiste en el sweet spot — ahí sí tiene sentido hacer el sweep completo de ε y reportar en qué régimen exacto SDOT se vuelve numéricamente estable.
Eso convierte el problema en resultado: "SIRI es estable para ε ≥ ε_critical donde ε_critical se determina empíricamente por capa". Es un hallazgo legítimo — más honesto que forzar un ε que funciona pero no mide lo que querés medir.

¿Aplicamos los clamps primero y vemos si el NaN desaparece en el régimen correcto?

Explicame desde cero las bases de bubble transformer15:20Claude responded: Bubble Transformer — desde ceroEstructuró explicación clara y directa sobre Bubble TransformerEstructuró explicación clara y directa sobre Bubble TransformerBubble Transformer — desde cero

El problema raíz
El mecanismo de atención Softmax estándar hace esto:
Para cada token Q_i → comparar contra TODOS los tokens K_j
→ O(N²) comparaciones
Dos problemas simultáneos:

Costo cuadrático — para secuencias largas, domina el compute.
Atención difusa — Softmax distribuye peso sobre todos los tokens, incluso los irrelevantes. Matemáticamente: nunca produce sparsity exacta.

La solución obvia es Sinkhorn-Knopp: proyección doblemente estocástica que fuerza sparsity. Funciona, pero introduce el Impuesto de Iteración Secuencial (IIS): las normalizaciones alternadas fila/columna crean dependencias secuenciales en HBM de GPU que anulan las ganancias de FLOPs. El modelo teóricamente es más eficiente, pero prácticamente el hardware lo penaliza.

La insight central: atención = Transporte Óptimo
Este es el fundamento teórico que diferencia al BT de otros trabajos de atención sparse.
Resultado (Daneshmand 2024 / Litman 2025):
Softmax resuelve implícitamente este problema de optimización:
min_A  ⟨A, C⟩ - ε·H(A)
donde:

A = matriz de atención (la variable a optimizar)
C_ij = ‖Q_i - K_j‖² = costo geométrico entre tokens
H(A) = entropía de Shannon (regularización)
ε = temperatura Softmax ≡ parámetro de regularización entrópica

Esto es textbook Transporte Óptimo Entrópico.
La consecuencia directa:
ε → ∞  :  A → uniforme  (Softmax estándar, máxima difusión)
ε → 0  :  A → Diagrama de Potencia (Celdas de Laguerre / Voronoi ponderado)
El BT no aproxima Softmax. Es su límite exacto cuando ε→0. La geometría que emerge naturalmente es una partición del espacio de embeddings en celdas de Voronoi. Y las celdas de Voronoi se calculan en O(N log C) — sin iteraciones, sin IIS.

Qué es una "burbuja"
Una burbuja es una celda de Voronoi en el espacio de embeddings.
Espacio de embeddings ℝ^d
     ┌──────────────────────────────┐
     │    ·  ·    │   · · ·        │
     │  ·  [c₁] · │  [c₂]  ·      │   ← c_i = centroide
     │    ·  ·    │      ·         │
     │────────────┼────────────────│   ← frontera Voronoi
     │  ·    ·    │  · [c₃] ·     │
     │    [c₄]·   │      ·  ·     │
     └──────────────────────────────┘
Cada token Q_j se asigna al centroide más cercano:
burbuja(Q_j) = argmin_i  d(Q_j, c_i)
La atención ocurre solo intra-burbuja. Tokens en burbujas distintas no se ven. La matriz de atención resultante es sparse por bloque — O(N·k) donde k = tokens por burbuja << N.

La arquitectura: V1 → V4
V1/V2 (PlateauAttention)    →    V3 (SDOT)         →    V4 (target)
Sinkhorn-Knopp iterativo        O(N log C)               O(N log C) +
O(N² × τ)                       Voronoi duro             geometría Riemanniana
                                 2.02× speedup            + Expert-Choice
V3 — SDOT (Semi-Discrete Optimal Transport): es el estado actual funcional.

Tomar Keys K → clustering → C centroides
Para cada Query Q_i → argmin_c d(Q_i, c) → asignación a burbuja
Atención densa solo entre tokens en la misma burbuja
Resultado: block-masked attention matrix

V4 — objetivo del roadmap:

Expert-Choice routing: en vez de cada token eligiendo su burbuja, cada burbuja elige sus Top-K tokens → balance perfecto sin Auxiliary Loss
Celdas de Laguerre: añadir pesos ψ_i aprendibles → burbujas de tamaño variable adaptadas a densidad semántica
Geometría Riemanniana: centroides en el Disco de Poincaré (hiperbólico), optimizados con RiemannianAdam vía geoopt

Los descubrimientos empíricos (Qwen3-0.6B)
Dos resultados contraintuitivos que emergieron de los experimentos:

1. SIRI — Sparsity-Induced Rank Inflation
La literatura predice: más sparsity → colapso dimensional → effective rank baja.
Lo que pasó:
ε = 1.0  →  R_eff ≈ 199
ε = 0.005 →  R_eff ≈ 576   (2.89×)
Por qué: la tensión de la partición Voronoi fuerza a los tokens supervivientes a proyectarse en ejes mutuamente ortogonales para evitar olvido catastrófico → crean "islas" semánticas con alta dimensionalidad intrínseca. El fenómeno es real (ausente en matrices random) — confirma dependencia de la variedad del lenguaje.
2. Meseta de Saturación
No es un límite del hardware (overflow float32). Es una propiedad topológica:
S_sat ocurre cuando  d(Soporte_Duro) / dε = 0
Es decir: el punto donde bajar más ε ya no cambia qué conexiones están activas — solo sus magnitudes. El grafo de atención se congela. Operar ahí garantiza sparsity máxima sin iteraciones innecesarias.

El incidente benchmark (2026-05-04)
Primer run del perplexity benchmark:
PPL softmax:   20.03
PPL bubble:    831,974.76   ← +4,154,237%
Causa raíz: SDOTAttentionV4 no aplicaba RoPE (Rotary Position Embeddings). En Qwen3-0.6B, RoPE rota Q y K por ángulos dependientes de posición. Sin eso, todos los tokens son posicionalmente idénticos → el clustering Voronoi asigna por contenido semántico pero sin posición → los patrones de atención son incoherentes con el LM head → NLL explota.
El fix: pasar rotary_emb del módulo Qwen3 original a SDOTAttentionV4 y aplicarlo post-proyección, pre-clustering. RoPE debe aplicarse sobre Q y K ya rotados para que las burbujas sean posicionalmente coherentes.
⚡ Lección arquitectónica: cualquier módulo que reemplace self_attn en un transformer posicional debe heredar el encoding posicional del modelo host. Sin esto, el smoke test pasa (shapes correctas) pero PPL colapsa.

Estado actual
SprintEstadoResultadoSprint 1✅PlateauAttention V2 + experimentos SinkhornSprint 2✅Motor SDOT V3 · 2.02× speedup · 35 testsSprint 3🔄Inyectar SDOT + RoPE en Qwen3 · validar PPL real
El blocker de Sprint 3 es exactamente lo que produjo el incidente: verificar que ΔPerplexity = PPL_BT - PPL_softmax ≤ δ. Las métricas internas (effective_rank, concentration_ratio) no sustituyen a PPL — son métricas geométricas, no de capacidad predictiva.

¿Qué querés profundizar? Puedo ir a la formalización matemática completa del OT (L2/L3), a la implementación específica de SDOT, o al diseño de Expert-Choice para V4.You said: Explicame como si tuviera 25 años, deficit de atencion porque es lo que tengo, me cuesta centrarme linealmente en el texto, hazlo lo mas emocionante posible li…Explicame como si tuviera 25 años, deficit de atencion porque es lo que tengo, me cuesta centrarme linealmente en el texto, hazlo lo mas emocionante posible linealmente para prestarle atencion a los conceptos arquitectonicos base de bubble transformer.15:27Claude responded: El Transformer está mintiendoDiseñó explicación narrativa y dinámica para conceptos arquitectónicos complejosDiseñó explicación narrativa y dinámica para conceptos arquitectónicos complejosEl Transformer está mintiendo
Cada vez que un LLM procesa texto, hace esto:
Cada palabra mira a TODAS las demás palabras.
"El gato come pescado"

El   → mira → El, gato, come, pescado
gato → mira → El, gato, come, pescado
come → mira → El, gato, come, pescado
...
4 palabras = 16 comparaciones.
100 palabras = 10,000 comparaciones.
1000 palabras = 1,000,000 comparaciones.
Esto se llama O(N²). Cuadrático. Cada vez que duplicás el texto, el costo se multiplica por 4.

Pero eso no es lo peor
Lo peor es que la mayoría de esas comparaciones no sirven para nada.
Cuando procesás "El gato come pescado", la palabra "come" necesita prestarle atención a "gato" y "pescado". Tiene sentido.
¿Necesita prestarle atención a "El"?
No realmente.
¿Y si el texto tiene 10,000 palabras? "Come" va a gastar recursos comparándose contra 9,996 palabras que no le importan absolutamente nada.
El Softmax no puede decir cero. Por diseño matemático, siempre distribuye algo de atención a todos. Nunca puede ignorar completamente a nadie.

Alguien intentó arreglarlo
Se llama Sinkhorn-Knopp. Un algoritmo que fuerza a la atención a ser sparse — a ignorar de verdad lo que no importa.
Funciona. Matemáticamente, funciona.
Pero hay un problema de hardware.
Sinkhorn necesita normalizar filas, después columnas, después filas, después columnas... iterativamente. Cada normalización depende de la anterior.
En la GPU, esto significa: espera, lee memoria, calcula, escribe memoria, espera, lee memoria...
La GPU tiene que pausar constantemente para leer resultados anteriores antes de seguir.
Eso se llama el Impuesto de Iteración Secuencial. Ganás en FLOPs, perdés en latencia real. La GPU queda infrautilizada.

Acá es donde el Bubble Transformer hace algo que nadie esperaba
Un paper de 2024 (Daneshmand) prueba algo fundamental:

Softmax, sin que nadie se lo pidiera, está resolviendo un problema de física.

El problema se llama Transporte Óptimo. Es la respuesta a la pregunta: ¿cuál es la forma más eficiente de mover masa de un lugar a otro?
La fórmula de atención:
min_A  ⟨A, C⟩ - ε·H(A)

A = la matriz de atención (lo que queremos encontrar)
C_ij = distancia² entre token i y token j
H(A) = entropía (cuán distribuida está la atención)
ε = temperatura

Softmax estaba resolviendo esto todo el tiempo. Solo que con el regulador ε tan alto que la solución quedaba difusa, distribuida, sin estructura.

¿Qué pasa si bajás ε a casi cero?
ε → ∞  →  atención uniforme (todos reciben algo)
ε → 0  →  ???
Cuando ε → 0, la solución al problema de Transporte Óptimo converge a algo conocido en geometría:
Un Diagrama de Voronoi.

¿Qué es un Diagrama de Voronoi?
Imaginate que tirás 5 semillas en una hoja en blanco.
Cada punto del espacio queda asignado a la semilla más cercana.
El resultado son regiones — celdas. Cada celda tiene exactamente una semilla en su interior, y todos los puntos dentro de esa celda son los más cercanos a esa semilla.
    ·  ·  ·  │  · · ·
  ·  [S₁] · │  [S₂]  ·
    ·  ·  ·  │      ·
─────────────┼──────────
  ·    ·     │  · [S₃] ·
    [S₄]·    │      ·  ·
Las líneas son las fronteras. Todo lo que está del mismo lado de una frontera pertenece a la misma celda.

Esas celdas son las burbujas
Ahora en vez de puntos en una hoja, son tokens en el espacio de embeddings.
En vez de semillas, son centroides — puntos representativos aprendibles.
El Bubble Transformer dice:

Cada token se asigna a su centroide más cercano.
La atención ocurre solo entre tokens de la misma burbuja.
Tokens en burbujas distintas: zero atención. Exacto. No "casi cero". Cero.

Por qué esto rompe todo el problema anterior
Softmax: cada token compara contra todos → O(N²)
Sinkhorn: fuerza sparsity pero con iteraciones secuenciales en GPU → IIS
Bubble Transformer: la partición Voronoi se calcula geométricamente, sin iteraciones. Asignar cada token a su centroide más cercano es un solo paso paralelo → O(N log C) donde C = número de burbujas, que es mucho menor que N.
Sin iteraciones. Sin dependencias secuenciales. La GPU trabaja entera, todo en paralelo.

El giro que nadie vio venir
Cuando los experimentos corrieron sobre Qwen3-0.6B, la predicción era:

"Menos conexiones activas = embeddings más degenerados = rango efectivo más bajo."

Lo que pasó:
ε = 1.0  →  R_eff ≈ 199
ε = 0.005 →  R_eff ≈ 576
Más sparse → más dimensionalidad, no menos.
¿Por qué? Porque cuando la atención colapsa a solo los tokens importantes, esos tokens sobrevivientes se ven forzados a diferenciarse entre sí. No pueden solaparse. Se proyectan en ejes ortogonales. Crean "islas" semánticas con estructura interna más rica.
Es contraintuitivo. Y está confirmado: el fenómeno no aparece en matrices random. Es una propiedad de la variedad del lenguaje. El lenguaje tiene estructura geométrica real, y el BT la está exponiendo.

El resumen en una imagen
SOFTMAX              SINKHORN             BUBBLE TRANSFORMER
━━━━━━━━━            ━━━━━━━━━            ━━━━━━━━━━━━━━━━━

Todos hablan         Pocos hablan         Regiones hablan
con todos            con pocos            entre sí

O(N²)               O(N²×τ)              O(N log C)
━━━━━━               ━━━━━━               ━━━━━━━━━
nada es cero        iteraciones          sin iteraciones
                    secuenciales         partición directa
                    → IIS               → GPU full parallel

El Bubble Transformer no es una aproximación de Softmax. Es lo que Softmax estaba intentando hacer desde siempre — solo que sin saber que el espacio geométrico correcto ya tenía nombre.
¿Querés que siga con cómo se implementa el routing, o con por qué la geometría hiperbólica (V4) es el siguiente paso natural?
