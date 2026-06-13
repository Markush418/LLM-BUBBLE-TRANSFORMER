# META-PROMPT: LLM-BUBBLE SPRINT 4 — ENTRENAMIENTO NATIVO SDOT DESDE CERO

## ⚠ CONTEXTO CRÍTICO — LEER PRIMERO

Sprint 3 probó SDOT inyectado sobre Qwen3-0.6B ya entrenado con Softmax.
Resultado: +0.90% PPL con soft routing. Válido pero incompleto.

Por qué Sprint 4 es el experimento que realmente valida el Bubble Transformer:

Un modelo pre-entrenado con Softmax tiene todos sus pesos optimizados
asumiendo atención densa. Inyectar SDOT post-hoc es como cambiar las
reglas del juego después de que el modelo aprendió a jugar. El resultado
(+0.90%) es el costo de esa incompatibilidad arquitectónica — no el
rendimiento real de SDOT.

Sprint 4 entrena un modelo PEQUEÑO desde cero donde SDOT es el mecanismo
nativo desde el primer paso de gradiente. Si PPL_SDOT ≈ PPL_Softmax con
el mismo número de parámetros y tokens vistos → el BT está validado.

---

## CONTEXT LOCK — Estado actual del repo

Lo que existe y es reutilizable:

1. `models/qwen3_gqa_bubble_wrapper.py` — wrapper GQA-native completo y funcional
2. `models/sdot_attention_v4.py` — módulo SDOT V3 con soft routing + causal mask
3. `v4_core.py` — Expert-Choice routing con clamps NaN (WIP, tests pasando)
4. `metrics.py` — 5 métricas de concentración incluyendo R_eff (SIRI)
5. `perplexity_benchmark.py` — benchmark WikiText-2 sliding window validado
6. `siri_ppl_results.json` — resultados históricos de todos los sprints

Lo que NO existe y debe crearse en Sprint 4:

1. Arquitectura nanoGPT-SDOT: modelo tiny con SDOT nativo (no wrapper)
2. Training loop con AdamW, LR scheduler, gradient clipping
3. Dual training: misma semilla, mismos datos, Softmax vs SDOT en paralelo
4. Comparación final: PPL_SDOT vs PPL_Softmax bajo condiciones idénticas

---

## SOURCE OF TRUTH — Hardware y Constraints

| Parámetro        | Valor                              |
|------------------|------------------------------------|
| GPU              | GTX 1650, 4GB VRAM                 |
| dtype            | float16 ONLY (no bfloat16)         |
| Max batch VRAM   | batch_size=4, seq_len=256          |
| Framework        | PyTorch + transformers             |
| Dataset          | WikiText-2 (ya descargado y usado) |
| Python           | 3.10+                              |
| Objetivo PPL     | ≤ PPL_Softmax + 2%                 |

### Arquitectura del modelo nativo (NanoGPT-SDOT)

El modelo de referencia es NanoGPT reducido para caber en 4GB con training:

| Parámetro       | Valor    | Razón                                     |
|-----------------|----------|-------------------------------------------|
| n_layers        | 6        | Suficiente para mostrar deep layer effect |
| n_heads         | 8        | MHA pura (no GQA) — más simple            |
| head_dim        | 64       | d_model / n_heads                         |
| d_model         | 512      | ~25M params total — cabe en 4GB           |
| d_ffn           | 2048     | 4 × d_model                               |
| seq_len         | 256      | Cabe con batch_size=4 en float16          |
| vocab_size      | 50257    | GPT-2 tokenizer (ya disponible en HF)    |
| num_bubbles (C) | 32       | NUM_BUBBLES irrelevante per Sprint 3      |
| routing_bonus   | 0.2      | Sweet spot confirmado en Sprint 3         |
| tuning          | entropy-weighted | Reduce PPL -50% vs uniform        |

⚠ NO usar GQA en el modelo nativo. Sprint 3 ya resolvió GQA-compatibility.
Sprint 4 valida SDOT en su forma más limpia: MHA pura, sin complejidad extra.

⚠ NO empezar con modelos grandes (Qwen3, Llama). El objetivo es una comparación
controlada. Un modelo de 25M parámetros con WikiText-2 es suficiente para
publicar el resultado de validación.

---

## PLAN DE EJECUCIÓN — 4 FASES SECUENCIALES

### FASE 1 — Arquitectura NanoGPT-SDOT (crear desde cero)

Crear `models/nano_sdot_gpt.py`:

```python
class NanoSDOTGPT(nn.Module):
    """
    GPT minimal con SDOT como mecanismo nativo de atención.
    NO es un wrapper — SDOT está hardcodeado en cada capa.
    """
    def __init__(self, config: NanoConfig):
        # config.attention_type ∈ {"softmax", "sdot"}
        # Misma arquitectura, distinto mecanismo de atención
        # Permite entrenamiento dual con misma semilla

class NanoConfig:
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ffn: int = 2048
    seq_len: int = 256
    vocab_size: int = 50257
    dropout: float = 0.1
    num_bubbles: int = 32       # solo para SDOT
    routing_bonus: float = 0.2  # solo para SDOT
    bonus_tuning: str = "entropy_weighted"  # o "uniform"
    attention_type: str = "softmax"  # o "sdot"
```

La capa de atención SDOT nativa (NO wrapper):

```python
class SDOTAttentionNative(nn.Module):
    """
    SDOT sin wrapper — atención nativa para modelo trained from scratch.
    Sin GQA. Sin position_embeddings externos. Sin rotary_emb del host.
    Usa learned positional embeddings estándar del NanoGPT.
    """
    def __init__(self, config):
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.centroids = nn.Parameter(
            torch.randn(num_bubbles, head_dim) * 0.02
        )
        self.scaling = head_dim ** -0.5

    def forward(self, x, attn_mask=None):
        # PASO 1: proyectar
        # PASO 2: reshape a [B, heads, N, head_dim]
        # PASO 3: soft routing — calcular assignments + bonus
        # PASO 4: atención densa + causal mask
        # PASO 5: merge + output projection
```

⚠ En el modelo nativo NO hay RoPE. Los positional embeddings son learned
(suma al input embedding, estándar NanoGPT). Esto simplifica la
implementación y es correcto para un modelo entrenado desde cero.

---

### FASE 2 — Training Loop Dual

Crear `train_dual.py`:

```python
"""
Entrena DOS modelos en paralelo con exactamente los mismos:
- Seed aleatoria
- Datos (mismo orden de batches)
- Hiperparámetros de optimización
- Número de pasos

La única diferencia: attention_type = "softmax" vs "sdot"
Esto controla todas las variables excepto el mecanismo de atención.
"""

TRAINING_CONFIG = {
    "n_steps": 5000,          # ~2h en GTX 1650
    "batch_size": 4,
    "seq_len": 256,
    "lr": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 200,
    "eval_interval": 500,     # evaluar PPL cada 500 steps
    "seed": 42,
    "dtype": torch.float16,
    "device": "cuda",
    "dataset": "wikitext-2-raw-v1",
    "tokenizer": "gpt2",      # GPT-2 tokenizer = 50257 vocab
}
```

El training loop debe:

1. Inicializar ambos modelos con la misma seed (`torch.manual_seed(42)`)
2. Crear UN solo DataLoader — ambos modelos ven los mismos batches en orden
3. Optimizador AdamW independiente para cada modelo
4. LR scheduler cosine con warmup para ambos
5. Cada `eval_interval` steps: calcular PPL en WikiText-2 test para ambos
6. Guardar resultados en `sprint4_training_results.json` con formato:

```json
{
  "step": 500,
  "softmax_train_loss": 4.21,
  "sdot_train_loss": 4.18,
  "softmax_ppl_eval": 89.3,
  "sdot_ppl_eval": 91.2,
  "sdot_delta_pct": "+2.1%"
}
```

---

### FASE 3 — Métricas SIRI durante training

Crear `scripts/siri_during_training.py`:

Cada `eval_interval` steps, además del PPL, medir R_eff en los embeddings
de salida de atención de cada capa para el modelo SDOT:

```python
# Para cada capa l en [0, n_layers-1]:
#   Extraer embeddings post-atención para un batch de evaluación fijo
#   Calcular R_eff = exp(H(σ/||σ||₁))
#   Guardar (step, layer, r_eff)
```

Esto permite responder: ¿SIRI emerge durante el training nativo, o solo
era un artefacto del wrapper sobre Qwen3?

Si SIRI emerge durante training nativo → el fenómeno es propiedada del
mecanismo SDOT, no de Qwen3 específicamente → fortalece el paper.

---

### FASE 4 — Comparación Final y Visualización

Crear `scripts/sprint4_analysis.py`:

Tabla de resultados final:

```
Modelo              | PPL Final | Params | Steps | Time
────────────────────|-----------|--------|-------|──────
NanoGPT (Softmax)   |    ???    |  25M   | 5000  |  ~?h
NanoGPT (SDOT)      |    ???    |  25M   | 5000  |  ~?h
Delta               |    ???%   |   =    |   =   |  ~?h
```

Criterio de éxito:

- ✅ PPL_SDOT ≤ PPL_Softmax × 1.02 → BT validado como mecanismo nativo
- ⚠ PPL_SDOT ≤ PPL_Softmax × 1.05 → viable con tuning adicional
- ❌ PPL_SDOT > PPL_Softmax × 1.05 → requiere diagnóstico (ver §Troubleshooting)

Además graficar (usar matplotlib, guardar como PNG):

1. `sprint4_loss_curves.png` — train loss de ambos modelos por step
2. `sprint4_ppl_curves.png` — PPL eval de ambos modelos por eval checkpoint
3. `sprint4_siri_evolution.png` — R_eff por capa a lo largo del training SDOT

---

## FEATURE BLOCKS

### FEATURE 1 — NanoConfig + NanoSDOTGPT (MANDATORY)

Crear `models/nano_sdot_gpt.py` con:

- `NanoConfig` dataclass con todos los parámetros listados
- `NanoSDOTGPT(nn.Module)` con `attention_type` como parámetro
- `SDOTAttentionNative` — atención SDOT sin wrapper, sin GQA, sin RoPE externo
- `SoftmaxAttentionNative` — atención MHA estándar (misma interfaz)
- Causal mask precalculada en `__init__` y registrada como buffer

### FEATURE 2 — train_dual.py (MANDATORY)

Script de training dual con:

- Seed sincronizada entre los dos modelos
- DataLoader compartido (mismo orden de batches para ambos)
- Logging a `sprint4_training_results.json` cada eval_interval
- Checkpoint saving a `checkpoints/softmax_stepN.pt` y `checkpoints/sdot_stepN.pt`
- Progress bar (tqdm) mostrando ambas losses en tiempo real
- Manejo de OOM: si CUDA OOM → reducir batch_size automáticamente e intentar de nuevo

### FEATURE 3 — siri_during_training hook (HIGH PRIORITY)

Hook que se activa en cada eval checkpoint:

- Extrae embeddings de cada capa del modelo SDOT
- Calcula R_eff con la implementación de `metrics.py`
- Acumula resultados en `sprint4_siri_evolution.json`

### FEATURE 4 — sprint4_analysis.py (HIGH PRIORITY)

Script de análisis post-training:

- Lee `sprint4_training_results.json`
- Genera las 3 figuras matplotlib
- Imprime tabla de resultados final con veredicto automático (✅/⚠/❌)
- Actualiza `siri_ppl_results.json` con los resultados de Sprint 4

### FEATURE 5 — Smoke test pre-training (MANDATORY)

ANTES de lanzar 5000 steps, ejecutar smoke test de 10 steps con ambos modelos:

- Verificar que loss baja (no NaN, no Inf, no plateau inmediato)
- Verificar shapes de todos los tensores intermedios
- Estimar tiempo total de training (extrapolación de 10 steps)
- Si estimación > 4h → reducir n_steps o n_layers y re-estimar

---

## TECHNICAL CONSTRAINTS

- Stack: torch, transformers (tokenizer only), tqdm, matplotlib, numpy, json
- NO nuevas librerías — todo lo anterior ya está en el entorno
- float16 ONLY — GTX 1650 no soporta bfloat16
- gradient checkpointing: usar si VRAM < 500MB libre durante training
- `torch.compile()`: NO usar — requiere triton, no disponible en GTX 1650
- Seed: SIEMPRE `torch.manual_seed(42)` + `torch.cuda.manual_seed(42)` antes de inicializar modelos
- Centroids en SDOT nativo: `nn.Parameter` con `requires_grad=True` — los centroides SE ENTRENAN junto con el resto del modelo

### Sobre los centroides aprendibles

En el wrapper de Sprint 3, los centroides eran fijos (no se entrenaban porque
el modelo base Qwen3 era frozen). En Sprint 4, los centroides son parámetros
aprendibles — forman parte del grafo computacional. El gradient fluye a través
de la asignación Voronoi vía el routing_bonus (soft routing es diferenciable).
La asignación dura (argmin) NO es diferenciable — NO backpropagar a través de ella.

---

## OUTPUT STRUCTURE REQUIRED

1. `models/nano_sdot_gpt.py` — NUEVO — NanoConfig + NanoSDOTGPT + ambas capas de atención
2. `train_dual.py` — NUEVO — training loop dual con logging y checkpoints
3. `scripts/siri_during_training.py` — NUEVO — hook SIRI durante training
4. `scripts/sprint4_analysis.py` — NUEVO — análisis + figuras + tabla final
5. `sprint4_training_results.json` — GENERADO por train_dual.py durante ejecución
6. `sprint4_siri_evolution.json` — GENERADO por hook SIRI durante ejecución
7. `siri_ppl_results.json` — ACTUALIZADO con resultado final de Sprint 4
8. `checkpoints/` — directorio creado automáticamente por train_dual.py

---

## QUALITY GATES — NO SUBMIT SIN PASAR

- [ ] `NanoSDOTGPT(config)` con `attention_type="softmax"` y `attention_type="sdot"` ambos instancian sin error
- [ ] Ambos modelos tienen exactamente el mismo número de parámetros (excepto `centroids` en SDOT: +num_bubbles×head_dim params)
- [ ] Smoke test 10 steps: loss baja en ambos modelos (no NaN, no plateau)
- [ ] Causal mask verificada: `mask[i,j] = -inf` para j > i en todos los heads
- [ ] `centroids.requires_grad = True` en modelo SDOT
- [ ] DataLoader seed sincronizada: batch 0 idéntico en ambos modelos (verificar con `assert torch.equal(batch_softmax, batch_sdot)`)
- [ ] `sprint4_training_results.json` existe y tiene al menos 1 entrada después de 500 steps
- [ ] Checkpoint saved en `checkpoints/softmax_step500.pt` y `checkpoints/sdot_step500.pt`
- [ ] `sprint4_analysis.py` genera las 3 figuras PNG sin error
- [ ] Tabla final impresa con veredicto (✅/⚠/❌) basado en el threshold del 2%

---

## §TROUBLESHOOTING — Si PPL_SDOT >> PPL_Softmax

Si después de 5000 steps PPL_SDOT > PPL_Softmax × 1.05, diagnosticar en este orden:

**1. Verificar que los centroides están aprendiendo**

```python
# Agregar al final de cada eval checkpoint:
print(f"Centroid std: {model_sdot.centroids.std():.4f}")
# Si std < 0.01 → centroides colapsados → el routing_bonus es demasiado bajo
```

**2. Verificar distribución de asignaciones**

```python
# Para un batch de eval, imprimir distribución de tokens por burbuja:
# Si 1-2 burbujas tienen >80% de los tokens → colapso → subir LR de centroides
```

**3. Probar routing_bonus más alto**

```
routing_bonus = 0.5 → si mejora, el modelo necesita señal de clustering más fuerte
```

**4. Verificar gradient flow a centroides**

```python
# Después de backward():
print(f"Centroid grad norm: {model_sdot.centroids.grad.norm():.4f}")
# Si grad = 0 → gradiente no fluye → revisar que routing_bonus está en el grafo
```

---

## INTERPRETACIÓN DE RESULTADOS — GUÍA PARA EL PAPER

### Si PPL_SDOT ≤ PPL_Softmax × 1.02

Resultado: **BT VALIDADO COMO MECANISMO NATIVO**
Narrative del paper: "Under native training conditions, SDOT attention matches
Softmax perplexity within 2% while achieving O(N log C) theoretical complexity."

### Si PPL_SDOT entre 1.02 y 1.05

Resultado: **VIABLE CON TUNING ADICIONAL**
Narrative: "Native SDOT training achieves within 5% PPL of Softmax at 25M params,
suggesting hyperparameter sensitivity that warrants further investigation at scale."

### Si PPL_SDOT > 1.05

Resultado: **REQUIERE AJUSTE ARQUITECTÓNICO**
Narrative (honesto para el paper): "Hard routing in native training exceeds
acceptable PPL thresholds, motivating the Expert-Choice V4 architecture where
load balance is enforced without hard assignment."
→ En este caso, Sprint 4 se convierte en la motivación empírica para V4.

---

## RELACIÓN CON PAPER arXiv

Los resultados de Sprint 4 van en §5 "Empirical Findings" como subsección nueva:

```
5.5 Native Training Validation
"We train NanoGPT-SDOT (25M params, 6 layers) from scratch on WikiText-2
for 5,000 steps under identical conditions as a Softmax baseline.
Final PPL: Softmax=X.XX, SDOT=X.XX (ΔX.XX%).
SIRI emergence during training: R_eff increases from Y to Z across layers..."
```

El paper actualmente tiene [CITATION: Sprint 4 pending] en la sección
de Future Work — Sprint 4 convierte ese placeholder en datos reales.
