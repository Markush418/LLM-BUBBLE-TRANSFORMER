# META-PROMPT: LLM-BUBBLE — SDOT GQA-NATIVE INTEGRATION FIX

## ⚠ CORRECCIÓN CONCEPTUAL CRÍTICA — LEER ANTES DE TOCAR CÓDIGO

El bug raíz NO es el valor de ε. El bug raíz es arquitectónico:
`SDOTAttentionV4` asume MHA (Multi-Head Attention) pero Qwen3-0.6B usa GQA.

La evidencia:
- ε=0.005 → PPL 7,141,188
- ε=0.1   → PPL 11,323,982 (PEOR con ε más alto)

En un sistema funcionando, ε más alto = atención más uniforme = PPL más cercana
a baseline. Que empeoró confirma que el forward pass está roto independientemente
de ε. El problema es el diseño del wrapper, no los hiperparámetros.

Decisión de arquitectura que cambia todo:
NO copiar pesos. NO hacer reshape. En cambio:
→ conservar q_proj/k_proj/v_proj/o_proj originales de Qwen3 intactos
→ SDOT reemplaza SOLO el cálculo de atención (post-proyección, pre-output)
→ GQA expansion (repeat_kv) se aplica antes del clustering

---

## CONTEXT LOCK

Problemas actuales en el codebase:

1. `SDOTAttentionV4.__init__` duplica pesos QKV de Qwen3 con shapes incompatibles.
   Qwen3 q_proj=[2048,1024] vs SDOT espera W_q=[1024,1024] → reshape incorrecto.

2. `SDOTAttentionV4.forward` NO aplica la causal mask (triangular inferior).
   Todos los tokens ven el futuro → logits incoherentes → PPL explota a millones.

3. GQA repeat_kv ausente: Qwen3 tiene 16 Q-heads y 8 KV-heads. Antes del
   clustering, K y V deben expandirse de [B,8,N,64] a [B,16,N,64].

4. RoPE aplicado incorrectamente (o ausente): debe correr post-proyección,
   pre-clustering, sobre Q y K ya en formato [B,heads,N,head_dim].

5. `swap_attention_layers` hace weight copy con reshape → eliminar completamente.
   El nuevo wrapper usa las proyecciones originales del modelo sin copiarlas.

6. TOP_K=1024 con seq_len=1024 desactiva el routing (burbuja = todo el contexto).
   Para validar PPL real, TOP_K debe ser << seq_len.

---

## SOURCE OF TRUTH — QWEN3-0.6B SPECS

| Parámetro       | Valor    | Nota                                    |
|-----------------|----------|-----------------------------------------|
| d_model         | 1024     |                                         |
| num_layers      | 28       |                                         |
| num_heads_q     | 16       | Q heads                                 |
| num_heads_kv    | 8        | KV heads (GQA ratio 2:1)                |
| head_dim        | 64       | d_model / num_heads_q = 1024/16         |
| q_proj shape    | [2048, 1024] | out=num_heads_q×head_dim×2 (Qwen3 internal) |
| k_proj shape    | [1024, 1024] | out=num_heads_kv×head_dim×2             |
| v_proj shape    | [1024, 1024] |                                         |
| o_proj shape    | [1024, 2048] | in=num_heads_q×head_dim×2               |
| RoPE module     | `layer.self_attn.rotary_emb` | firma: `(x, position_ids)` |
| DEVICE          | GTX 1650 | float16 only, no bfloat16               |
| VRAM            | 4GB      |                                         |

Parámetros SDOT para benchmark:
| Parámetro  | Valor recomendado | Razón                                       |
|------------|-------------------|---------------------------------------------|
| NUM_BUBBLES (C) | 32 o 64      | C << seq_len para routing real              |
| TOP_K      | 64 o 128          | tokens por burbuja << 1024 (activa routing) |
| EPS_STAR   | 0.005             | régimen SIRI confirmado                     |
| MAX_LENGTH | 512               | cabe en 4GB con float16                     |
| STRIDE     | 256               |                                             |

---

## FEATURE 1 — Qwen3GQABubbleWrapper (MANDATORY)

Crear `models/qwen3_gqa_bubble_wrapper.py` — nuevo módulo que envuelve
el cálculo de atención de Qwen3 sin tocar sus proyecciones.

```python
class Qwen3GQABubbleWrapper(nn.Module):
    """
    Reemplaza self_attn en un DecoderLayer de Qwen3.
    Conserva todas las proyecciones originales.
    Solo reemplaza el cálculo de attention scores.
    """
    def __init__(self, original_attn, num_bubbles=32, top_k=64, eps=0.005):
        super().__init__()
        # Mantener todas las proyecciones originales INTACTAS
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.rotary_emb = original_attn.rotary_emb
        
        # Config Qwen3
        self.num_heads = original_attn.num_heads          # 16
        self.num_kv_heads = original_attn.num_key_value_heads  # 8
        self.head_dim = original_attn.head_dim            # 64
        self.kv_groups = self.num_heads // self.num_kv_heads   # 2
        
        # SDOT params
        self.num_bubbles = num_bubbles
        self.top_k = top_k
        self.eps = eps
        
        # Centroides aprendibles en espacio de keys
        self.centroids = nn.Parameter(
            torch.randn(num_bubbles, self.head_dim) * 0.02
        )
```

El forward del wrapper debe seguir este pipeline EXACTO:

```
PASO 1: proyectar con los módulos originales
  Q = self.q_proj(hidden_states)  # [B, N, 2048]
  K = self.k_proj(hidden_states)  # [B, N, 1024]
  V = self.v_proj(hidden_states)  # [B, N, 1024]

PASO 2: reshape a [B, heads, N, head_dim]
  Q → [B, 16, N, 64]  (notar: 2048 = 16*64*2 → ver nota Qwen3)
  K → [B, 8, N, 64]
  V → [B, 8, N, 64]

PASO 3: aplicar RoPE (OBLIGATORIO)
  cos, sin = self.rotary_emb(hidden_states, position_ids)
  Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

PASO 4: GQA expansion (repeat_kv)
  K = K.repeat_interleave(self.kv_groups, dim=1)  # [B, 16, N, 64]
  V = V.repeat_interleave(self.kv_groups, dim=1)  # [B, 16, N, 64]

PASO 5: SDOT clustering sobre K
  K_flat = K.mean(dim=1)  # [B, N, 64] promedio entre heads para clustering
  dists = torch.cdist(K_flat, self.centroids.unsqueeze(0).expand(B,-1,-1))
  assignments = dists.argmin(dim=-1)  # [B, N] → burbuja de cada token

PASO 6: block-sparse attention con causal mask
  Para cada burbuja b:
    mask_b = (assignments == b)  # [B, N] tokens en esta burbuja
    Por cada par (i, j) en la burbuja:
      si j > i (futuro): score = -inf  ← CAUSAL MASK OBLIGATORIA
    attn_weights = softmax(Q_b @ K_b.T / sqrt(head_dim) + causal_mask)
    out_b = attn_weights @ V_b

PASO 7: merge y proyección de salida
  output = merge(out_b para cada burbuja)  # [B, N, d_model_internal]
  output = self.o_proj(output)              # [B, N, 1024]

PASO 8: return (output, None)  ← Qwen3 DecoderLayer espera exactamente 2 valores
```

⚠ NOTA sobre shape de Q en Qwen3:
q_proj tiene output 2048 = num_heads(16) × head_dim(64) × 2.
Esto es porque Qwen3 usa un internal expansion. Verificar con:
`assert Q.shape[-1] == self.num_heads * self.head_dim * 2`
Si confirma, hacer reshape a [B, N, 16, 128] y ajustar head_dim a 128 para Q.
Verificar en el modelo real antes de hardcodear.

---

## FEATURE 2 — swap_attention_layers refactorizado (MANDATORY)

Eliminar toda la lógica de weight copy/reshape del swap actual.

Nuevo swap:
```python
def swap_attention_layers(model, num_bubbles=32, top_k=64, eps=0.005,
                           target_layers=None):
    swapped = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if target_layers is not None and layer_idx not in target_layers:
            continue
        original_attn = layer.self_attn
        wrapper = Qwen3GQABubbleWrapper(
            original_attn=original_attn,
            num_bubbles=num_bubbles,
            top_k=top_k,
            eps=eps
        )
        layer.self_attn = wrapper
        swapped += 1
    return swapped
```

No hay weight copy. No hay reshape. Solo wrapper.

---

## FEATURE 3 — Causal mask helper (MANDATORY)

Implementar en `models/qwen3_gqa_bubble_wrapper.py`:

```python
def _make_causal_mask(seq_len, device, dtype):
    """Upper triangle = -inf, diagonal + lower = 0"""
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask  # [N, N]
```

Esta máscara DEBE aplicarse en PASO 6 antes del softmax.
Sin esta máscara, token[i] ve tokens[i+1..N] → PPL explota.

---

## FEATURE 4 — perplexity_benchmark.py actualizado (HIGH PRIORITY)

Cambios requeridos:
```python
# Parámetros correctos para routing real
NUM_BUBBLES = 32      # C << seq_len
TOP_K       = 64      # tokens por burbuja << seq_len
EPS_STAR    = 0.005   # régimen SIRI
MAX_LENGTH  = 512
STRIDE      = 256

# Swap usando nuevo wrapper
from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper
n_swapped = swap_attention_layers(model, num_bubbles=NUM_BUBBLES,
                                   top_k=TOP_K, eps=EPS_STAR)
print(f"Swapped {n_swapped} layers")

# Smoke test ANTES del benchmark completo
# Pasar 1 batch de 10 tokens, verificar que PPL < 1000
# Si smoke test falla → detener y reportar traceback completo
```

Agregar al output de resultados:
```
NUM_BUBBLES: {NUM_BUBBLES}
TOP_K:       {TOP_K}
EPS_STAR:    {EPS_STAR}
```

---

## FEATURE 5 — Debug mode para forward pass (HIGH PRIORITY)

Agregar flag `debug=True` al primer run. En modo debug, el forward del wrapper
debe imprimir para el primer batch:

```python
if self.debug and self._debug_count == 0:
    print(f"Q shape post-proj: {Q.shape}")
    print(f"K shape post-proj: {K.shape}")
    print(f"Q shape post-RoPE: {Q.shape}")
    print(f"K shape post-expand: {K.shape}")
    print(f"assignments unique bubbles: {assignments.unique().shape[0]}")
    print(f"causal_mask sample [0:4,0:4]: {causal_mask[0:4,0:4]}")
    print(f"output shape: {output.shape}")
    print(f"output has NaN: {output.isnan().any()}")
    self._debug_count += 1
```

Esto es MANDATORIO para diagnosticar si el fix funciona antes del benchmark completo.

---

## TECHNICAL CONSTRAINTS

- Stack: torch, transformers==4.51.0, numpy — sin nuevas librerías
- Python 3.10+ — type hints en todas las funciones nuevas
- float16 ONLY — GTX 1650 no soporta bfloat16
- VRAM budget 4GB — MAX_LENGTH=512 máximo para benchmark
- `apply_rotary_pos_emb` — importar de `transformers.models.qwen3.modeling_qwen3`
  o de `transformers.modeling_rope_utils` según versión 4.51.0
- NO usar F.scaled_dot_product_attention para el bloque sparse
  (no soporta block-sparse masks de forma eficiente en esta versión)
- El wrapper DEBE ser compatible con `model.generate()` — no solo forward pass
- Return siempre `(output, None)` — Qwen3 DecoderLayer hace unpacking de 2 valores

---

## PLAN DE EJECUCIÓN (ejecutar en orden)

### FASE 1 — Inspección (ANTES de escribir código)
```
1. Leer models/sdot_attention_v4.py completo
2. Leer perplexity_benchmark.py completo
3. Ejecutar este snippet para verificar shapes reales:
   
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", 
               torch_dtype=torch.float16)
   layer0 = model.model.layers[0].self_attn
   print("q_proj:", layer0.q_proj.weight.shape)
   print("k_proj:", layer0.k_proj.weight.shape)
   print("v_proj:", layer0.v_proj.weight.shape)
   print("o_proj:", layer0.o_proj.weight.shape)
   print("num_heads:", layer0.num_heads)
   print("num_kv_heads:", layer0.num_key_value_heads)
   print("head_dim:", layer0.head_dim)
   print("rotary_emb type:", type(layer0.rotary_emb))

4. Documentar los shapes reales — si difieren del SOURCE OF TRUTH, 
   ajustar ANTES de continuar.
```

### FASE 2 — Implementación
```
1. Crear models/qwen3_gqa_bubble_wrapper.py (Feature 1 + Feature 3)
2. Actualizar perplexity_benchmark.py (Feature 2 + Feature 4 + Feature 5)
3. NO modificar v4_core.py ni sdot_attention_v4.py en este sprint
```

### FASE 3 — Smoke test (checkpoint obligatorio)
```
Ejecutar con debug=True, solo 1 layer swapped, 10 tokens:
  swap_attention_layers(model, target_layers=[0], ...)

Criterio de paso:
  - No NaN en output
  - causal_mask[0:4,0:4] muestra -inf en upper triangle
  - PPL smoke < 1000  (si >= 1000 → reportar debug output y detener)
```

### FASE 4 — Benchmark completo
```
Solo ejecutar si Fase 3 pasa.
Reportar tabla:
  Modelo          | PPL
  Baseline        | ~25.72
  Bubble (C=32)   | ???
  
Si PPL Bubble < 500 → SIRI validation puede continuar
Si PPL Bubble > 1000 → reportar shapes del debug y abrir siguiente diagnóstico
```

---

## OUTPUT STRUCTURE REQUIRED

1. `models/qwen3_gqa_bubble_wrapper.py` — NUEVO — clase `Qwen3GQABubbleWrapper`
   con `__init__`, `forward`, `_make_causal_mask`, `_sdot_block_attention`

2. `perplexity_benchmark.py` — MEJORADO — swap refactorizado, parámetros
   corregidos, debug mode, smoke test pre-benchmark

3. `siri_ppl_results.json` — ACTUALIZADO — agregar entrada con nueva run:
   `{"run": "gqa_native_wrapper", "num_bubbles": 32, "top_k": 64, 
    "eps": 0.005, "ppl_baseline": X, "ppl_bubble": Y, "delta_pct": Z}`

4. `CHARLACLAUDE.md` o `TRESMAYAS.md` — ACTUALIZADO — documentar el 
   diagnóstico del bug GQA + solución arquitectónica adoptada

---

## QUALITY GATES — NO SUBMIT SIN PASAR

- [ ] `Qwen3GQABubbleWrapper` importa sin error desde `models/qwen3_gqa_bubble_wrapper.py`
- [ ] `layer0.q_proj` en el wrapper apunta al mismo objeto que el original (no copia)
- [ ] `_make_causal_mask(4, device, dtype)[0:4,0:4]` tiene -inf en posiciones (0,1),(0,2),(0,3)
- [ ] `swap_attention_layers` no ejecuta ningún `.copy_()` ni `reshape` de pesos
- [ ] Debug mode imprime shapes sin NaN para el primer batch
- [ ] Smoke test (1 layer, 10 tokens) pasa con PPL < 1000
- [ ] `return (output, None)` — exactamente 2 valores en forward
- [ ] RoPE se aplica con la firma `rotary_emb(hidden_states, position_ids)` 
      (no `rotary_emb(position_ids)`)
- [ ] `repeat_interleave(self.kv_groups, dim=1)` presente antes del clustering
- [ ] `siri_ppl_results.json` actualizado con resultados de la nueva run

---

## STYLE PRESERVATION

Mantener:
- Estructura de directorios existente (models/, benchmark en root)
- Convención de nombres snake_case
- Comentarios en español en código de orquestación
- Comentarios técnicos en inglés en módulos de ML

NO modificar:
- `v4_core.py` — contiene Expert-Choice routing de V4, no relacionado a este fix
- `metrics.py` — métricas de concentración, independientes del wrapper
- `plateau_attention.py` — V2 legacy, no tocar
