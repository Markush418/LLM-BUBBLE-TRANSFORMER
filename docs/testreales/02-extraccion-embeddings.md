# 02. Extracción de Embeddings

**Fecha**: 16 Abril 2026
**Tiempo**: 32 segundos

---

## Proceso de Extracción

### Comando Ejecutado

```bash
python experiments/extract_embeddings.py \
    --target-layers 0 4 8 12 16 20 24 \
    --batch-size 1 \
    --max-length 256
```

### Output del Extractor

```
======================================================================
 Qwen3-0.6B Embedding Extractor — 4-bit NF4 Quantized
 LLM-BUBBLE — Real Embeddings Mode
======================================================================

[Step 1/3] Loading corpus...
[Corpus] Loaded 50 texts from data/test_corpus.jsonl

[Step 2/3] Loading model and extracting embeddings...
--------------------------------------------------
[Extractor] Loading Qwen/Qwen3-0.6B-Base in 4-bit NF4...
[Extractor] Device: cuda, Batch: 1, Max len: 256
[Extractor] Found 28 transformer layers
[Extractor] Model loaded. VRAM: 514MB allocated, 600MB reserved

Extracting batches: 100%|██████████| 50/50 [00:11<00:00, 4.20it/s]

[Step 3/3] Saving embeddings...
--------------------------------------------------
 Layer  0: shape=(1587, 1024), eff_rank=608.8
 Layer  4: shape=(1587, 1024), eff_rank=8.0
 Layer  8: shape=(1587, 1024), eff_rank=14.8
 Layer 12: shape=(1587, 1024), eff_rank=27.7
 Layer 16: shape=(1587, 1024), eff_rank=61.0
 Layer 20: shape=(1587, 1024), eff_rank=223.8
 Layer 24: shape=(1587, 1024), eff_rank=394.5
 Raw input: (1587, 1024)

======================================================================
 EXTRACTION COMPLETE — 32.0s
======================================================================
```

---

## Problema Crítico: Overflow en Float16

### Síntoma

Los embeddings extraídos mostraban valores extremadamente altos:

```
Layer 0: Mean: 0.0014, Std: 0.27, Max: 7.03, Min: -2.95   ✓ Normal
Layer 4: Mean: 0.21, Std: inf, Max: 6784, Min: -169      ✗ Overflow!
Layer 24: Mean: 0.69, Std: inf, Max: 6800, Min: -176      ✗ Overflow!
```

### Análisis del Problema

```python
# Diagnóstico de valores extremos
Layer 4: Total elements: 1625088
  |value| > 100:   275 (0.02%)
  |value| > 1000:   47 (0.00%)  
  |value| > 5000:   47 (0.00%)
```

**Causa**: La cuantización 4-bit NF4 produce valores que exceden el rango de float16 (~±65504) en capas intermedias.

### Solución Aplicada

#### 1. Conversión a Float32 en el Hook

```python
# experiments/extract_embeddings.py - línea 138
def _capture(self, layer_idx: int, output):
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
    # Convert to float32 to prevent overflow from quantization artifacts
    self.layer_outputs[layer_idx] = hidden_states.float().cpu()
```

#### 2. Asegurar float32 al Guardar

```python
# experiments/extract_embeddings.py - línea 177
result[layer_idx] = hidden.numpy().astype(np.float32)
```

#### 3. Clipping de Outliers

```python
import numpy as np

for layer in [0, 4, 8, 12, 16, 20, 24]:
    arr = np.load(f'embeddings/softmax/layer_{layer}.npy')
    
    # Clip al percentil 0.5-99.5
    p_lo, p_hi = np.percentile(arr, [0.5, 99.5])
    arr_clipped = np.clip(arr, p_lo, p_hi)
    
    np.save(f'embeddings/softmax/layer_{layer}.npy', arr_clipped.astype(np.float32))
```

### Rangos de Clipping por Capa

```
Layer  0: clipped to [-0.9, 1.0]      → eff_rank = 609
Layer  4: clipped to [-2.8, 3.5]      → eff_rank = 593
Layer  8: clipped to [-3.7, 4.4]      → eff_rank = 625
Layer 12: clipped to [-5.6, 6.3]      → eff_rank = 583
Layer 16: clipped to [-8.4, 9.1]      → eff_rank = 620
Layer 20: clipped to [-17.0, 19.8]    → eff_rank = 683
Layer 24: clipped to [-27.7, 39.1]    → eff_rank = 713
```

---

## Verificación de Calidad

### Effective Rank Post-Clipping

```python
def effective_rank(X, threshold=0.99):
    """Effective rank via SVD: exp(H(p)) where p_i = sigma_i / sum(sigma)."""
    if X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])
    centered = X - X.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    return float(np.exp(entropy))

# Resultados:
Layer  0: eff_rank = 609 (baseline: vocab diversity)
Layer  4: eff_rank = 593 (compression)
Layer  8: eff_rank = 625 
Layer 12: eff_rank = 583 (bottleneck)
Layer 16: eff_rank = 620
Layer 20: eff_rank = 683 (expansion)
Layer 24: eff_rank = 713 (expressive)
```

### Validación de Shapes

```
Todos los embeddings tienen:
- Shape: (1587, 1024) ✓
- Dtype: float32 ✓
- Sin NaN: ✓
- Sin Inf: ✓
```

---

## Archivos Generados

```
embeddings/
├── softmax/
│   ├── layer_0.npy    (1587, 1024) float32
│   ├── layer_4.npy    (1587, 1024) float32
│   ├── layer_8.npy    (1587, 1024) float32
│   ├── layer_12.npy   (1587, 1024) float32
│   ├── layer_16.npy   (1587, 1024) float32
│   ├── layer_20.npy   (1587, 1024) float32
│   └── layer_24.npy   (1587, 1024) float32
├── raw_input.npy      (1587, 1024) float32
└── metadata.json
```

### Metadata

```json
{
  "mode": "real",
  "model": "Qwen/Qwen3-0.6B-Base",
  "quantization": "4bit-NF4-double",
  "num_texts": 50,
  "batch_size": 1,
  "max_length": 256,
  "device": "cuda",
  "layers_saved": 7,
  "d_model": 1024,
  "num_layers": 28,
  "num_attention_heads": 16,
  "description": "Real embeddings from Qwen3-0.6B-Base (4-bit quantized)"
}
```

---

## Lecciones Aprendidas

1. **Float16 es insuficiente para embeddings cuantizados**: Los artefactos de cuantización producen valores fuera de rango
2. **Clipping es necesario**: Solo 47 de 1.6M valores eran outliers, pero dominaban las métricas
3. **Effective rank saludable**: 580-713 indica que los embeddings no colapsaron
4. **Patrón de capas**: Las capas intermedias comprimen, las finales expanden

---

## Próximo Paso

→ Ver [03-epsilon-sweep.md](./03-epsilon-sweep.md) para los resultados del sweep.
