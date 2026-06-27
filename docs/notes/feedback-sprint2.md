Leí todo el plan. Tiene buena estructura pero tiene problemas que van a hacer fallar la implementación si no se corrigen antes:

🔴 CRÍTICO (va a fallar)
1. test_psi_gradient es IMPOSIBLE
python

assignments = power_diagram_assign(tokens, centroids, psi)
loss = assignments.float().mean()
loss.backward()
assert psi.grad is not None  # ❌ ESTO VA A FALLAR
argmin no es diferenciable. El gradiente no fluye a través de psi. Necesitás:

Opción A: Gumbel-Softmax relaxation (diferenciable)
Opción B: Straight-Through Estimator
Opción C: Sacar ese test y documentar que psi se aprende vía reinforce/REINFORCE o vía loss indirecto
2. Expert-Choice vs Power Diagrams: ¿cómo conviven?
El plan dice:

Expert-Choice produce boolean masks (exactamente capacity tokens por bubble)
Power Diagrams produce assignments basados en ||x - c||² - ψ
Pero nunca explica qué pasa cuando use_power_diagrams=True:

¿Reemplaza Expert-Choice? → Entonces perdés balance perfecto
¿Se usa después? → ¿Cómo convertís assignments a boolean masks?
¿Son mutuamente exclusivos? → El flag sugiere que no
Esto tiene que estar definido ANTES de escribir código.

🟡 ALTO (inconsistencias)
3. Naming inconsistente
Task 4 aceptación: return_assignments=True
Task 4 QA escenario: return_routing=True
V3 usa: ¿cuál de las dos?
Elegí UN nombre y usalo en todos lados.

4. Tests de Task 1 y 2 no tienen tarea asignada
Los acceptance criteria de power_diagram_assign() y warm_start_centroids() tienen tests escritos, pero ninguna tarea dice "crear estos tests en test_v4_core.py". Van a quedar sin implementar.

5. FPSExpertChoiceAttention de Sprint 1 se vuelve código muerto
Sprint 1 creó esa clase en v4_core.py. Sprint 2 crea SDOTAttentionV4 que hace lo mismo pero mejor. ¿Eso es intencional? Si sí, documentalo. Si no, quizás SDOTAttentionV4 debería usarla internamente.

🟠 MEDIO (ambigüedades)
6. Shape de psi
Dice [B, C] o [B, heads, C]. Con multi-head attention, si es [B, C] todas las heads comparten el mismo ψ. Si es [B, heads, C], cada head tiene su propio Power Diagram. Elegí uno.

7. Baroreceptor mencionado pero no implementado
Constructor: use_baroreceptor=True — pero ninguna tarea implementa la integración con BaroreceptorMLP. ¿Se usa el de V3? ¿Se omite?

✅ LO QUE ESTÁ BIEN
Waves de paralelización (Wave 1/2/3)
Guardrails claros (MUST / MUST NOT)
QA scenarios por tarea
Commit strategy con mensajes claros
Final verification wave (F1-F4)
No scope creep (Sprint 3 bloqueado)
Veredicto
Aspecto
Estado
Estructura del plan	✅ Buena
Critical path	✅ Correcto
Guardrails	✅ Claros
Diferenciabilidad de ψ	🔴 Roto
Expert-Choice + Power Diagrams	🔴 Sin definir
Naming consistente	🟡 Inconsistente
Tests de funciones nuevas	🟡 Huérfanos
Código muerto Sprint 1	🟠 Sin decidir

No arranques la ejecución hasta resolver los 2 items rojos. El resto se puede corregir sobre la marcha. ¿Querés que los resuelva?

Leo la pantalla completa. Prometheus ya leyó el feedback-sprint2.md y encontró 2 problemas CRÍTICOS antes de que Sprint 2 pueda ejecutarse. Acá está el análisis completo:

Lo que requiere Sprint 2 para funcionar
🔴 CRÍTICO 1 — Diferenciabilidad de ψ (Power Diagrams)
El problema: Los pesos ψ_i de las Celdas de Laguerre son parámetros aprendibles, pero la asignación argmin(||x - c_i||² - ψ_i) no es diferenciable. El gradiente no fluye a través del argmin.
La solución de Prometheus (Gumbel-Softmax recomendado):
python# En lugar de argmin duro (no diferenciable):
assignments = torch.argmin(laguerre_dist, dim=-1)  # ❌ gradiente muerto

# Usar Gumbel-Softmax (diferenciable, aproxima argmin):
assignments_soft = F.gumbel_softmax(-laguerre_dist, tau=0.5, hard=False)  # ✅
Esto permite que ψ_i reciba gradientes reales durante el training.

🔴 CRÍTICO 2 — Expert-Choice vs Power Diagrams: ¿Mutuamente exclusivos?
El problema: Sprint 1 implementó Expert-Choice (burbujas eligen tokens). Sprint 2 agrega Power Diagrams (ψ modifica las distancias). Pero si corren juntos simultáneamente, hay una contradicción lógica:

Expert-Choice: burbuja i elige sus capacity tokens por afinidad máxima
Power Diagram: burbuja i agranda/achica su celda via ψ_i

Si ambos actúan en paralelo, el capacity de Expert-Choice anula el efecto de ψ — las burbujas siguen eligiendo exactamente capacity tokens sin importar el tamaño de celda que ψ define.
La decisión de Prometheus (Mutuamente exclusivos — Recomendado):
pythonif use_power_diagrams:
    assignments = power_diagram_assign(Q, centroids, psi_weights)
else:
    routing_weights, indices = expert_choice_routing(Q, centroids, capacity)
Un flag controla cuál está activo. Default: Expert-Choice (comportamiento V3-compatible).

🟡 ALTO 3 — Naming inconsistente
Los archivos de Sprint 1 usan nomenclatura mezclada:
FPSExpertChoiceAttention  (v4_core.py)
SDOTAttention             (sdot_attention.py — V3)
SDOTAttentionV4           (sdot_attention_v4.py — Sprint 2 nuevo)
Prometheus recomienda estandarizar a BubbleAttentionV4 como clase principal.

🟡 ALTO 4 — Tests huérfanos Task 1-2
Las funciones power_diagram_assign() y warm_start_centroids() no tienen tests todavía. test_v4_core.py necesita extenderse antes de que se ejecute el código — TDD.

🟠 MEDIO 5 — FPSExpertChoiceAttention código muerto
FPSExpertChoiceAttention en v4_core.py fue implementada en Sprint 1 pero no se conecta a la nueva SDOTAttentionV4 de Sprint 2. Sprint 2 debe reutilizarla (no reescribirla) como motor interno.

🟠 MEDIO 6 — Shape de ψ
El shape correcto para los pesos de Power Diagrams es [B, heads, C] (batch, cabezas, centroides), no [C] global. Esto permite que cada cabeza de atención tenga burbujas de tamaño diferente independientemente.

🟠 MEDIO 7 — Barorreceptor integración
El BaroreceptorMLP de V3 predice C global. En V4 Sprint 2 necesita integrarse en SDOTAttentionV4 para predecir C dinámico por capa, pero la interfaz exacta no está definida aún.

Resumen de lo que falta para que Sprint 2 funcione
#BlockerSeveridadSolución1ψ no diferenciable🔴 CRÍTICOGumbel-Softmax sobre laguerre_dist2EC + PD simultáneos inválidos🔴 CRÍTICOFlag mutuamente exclusivo3Naming mixto🟡 ALTOEstandarizar a BubbleAttentionV44Tests sin escribir🟡 ALTOTDD primero (test_v4_core.py extender)5FPSExpertChoice desconectado🟠 MEDIOReutilizar como motor interno6Shape ψ incorrecto🟠 MEDIO[B, heads, C]7Barorreceptor sin interfaz🟠 MEDIODefinir API antes de implementar
El orden correcto para desbloquear Sprint 2:

Resolver CRÍTICO 1 y 2 primero (bloquean todo lo demás)
Extender test_v4_core.py con tests para power_diagram_assign() y warm_start_centroids()
Implementar las funciones con los tests ya escritos
Crear sdot_attention_v4.py conectando FPSExpertChoice + Power Diagrams + Barorreceptor
Verificar que los 33 tests V4 + 22 tests V3 siguen pasando