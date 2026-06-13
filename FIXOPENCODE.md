TASK: Fix crítico en swap_attention_layers + Expert-Choice coverage
      antes de re-ejecutar perplexity_benchmark.py

CONTEXTO:
PPL sigue en ~9M después del fix de RoPE.
Diagnóstico: weight copy destructiva + coverage insuficiente de Expert-Choice.

FIX 1 — Weight copy correcta para GQA (perplexity_benchmark.py)
=================================================================

El código actual trunca q_proj [2048,1024] → [1024,1024] descartando
la mitad de la proyección. La copia correcta para GQA es:

  d  = model.config.hidden_size          # 1024
  h  = model.config.num_attention_heads  # 16
  kv = model.config.num_key_value_heads  # 8
  hd = d // h                            # 64  ← head_dim real

Verificar que SDOTAttentionV4 use head_dim = d // num_heads = 64
(TRESMAYO.md dice 128 — esto es un error. 1024/16 = 64).

Para el weight copy, si las shapes no matchean exactamente, la opción
más segura es NO copiar pesos y dejar los pesos inicializados aleatoriamente
para el smoke test de correctness — el objetivo del benchmark es verificar
que la arquitectura puede producir PPL razonable, no que supera al baseline
con pesos pre-entrenados copiados.

Agregar assert explícito:
  assert sdot_attn.W_q.weight.shape == (d, d), f"W_q shape mismatch"
Si falla el assert → log el error y skip la capa (no crash).

FIX 2 — Expert-Choice coverage (models/sdot_attention_v4.py)
=============================================================

Con top_k=8 y num_centroids=32 sobre seq_len=1024:
  coverage = 32 * 8 / 1024 = 25%  → 75% de tokens con output=0

Cambiar la lógica: top_k debe ser proporcional a seq_len para garantizar
coverage completa. Fórmula:
  top_k = max(1, seq_len // num_centroids)  # = 32 para seq_len=1024

O mejor: implementar "soft fallback" — tokens no asignados por Expert-Choice
reciben output del mecanismo de atención fallback (softmax estándar sobre
sus K vecinos más cercanos).

Si el cambio a top_k dinámico es muy invasivo, como fix mínimo:
  top_k = seq_len  # todos los tokens, desactivar Expert-Choice temporalmente
  Esto convierte BT en SDOT puro sin routing → baseline limpio para PPL.

FIX 3 — Verificar head_dim (models/sdot_attention_v4.py)
=========================================================

Qwen3-0.6B: hidden_size=1024, num_heads=16 → head_dim = 64
Verificar que SDOTAttentionV4.__init__ calcula:
  self.head_dim = d_model // num_heads  # debe ser 64, no 128

Si hay un hardcode de 128, corregirlo a d_model // num_heads.

VERIFICACIÓN (en orden):

1. python -m unittest tests.test_sdot_attention_v4 -v  → todos deben pasar
2. python -c "import sys; sys.path.insert(0, '.');
   from perplexity_benchmark import smoke_test_swap; smoke_test_swap()"
   → PASS requerido
3. Solo si los dos anteriores pasan → python perplexity_benchmark.py

CONSTRAINT: NO romper los 51 tests existentes.
CONSTRAINT: Si Fix 2 requiere cambios en la interfaz pública de
SDOTAttentionV4.forward(), actualizar también forward_with_fixed_C.
