# 01. Configuración del Experimento

**Fecha**: 16 Abril 2026
**Duración**: ~10 minutos total

---

## Environment

```
Python:        3.14.3
PyTorch:       2.11.0+cu128
CUDA:          12.8
GPU:           NVIDIA GeForce GTX 1650 (4GB VRAM)
bitsandbytes:  0.45.3 (4-bit quantization)
transformers:  5.4.0
NumPy:         2.2.6
```

---

## Modelo: Qwen3-0.6B-Base

### Especificaciones Técnicas

| Parámetro | Valor |
|-----------|-------|
| Nombre | Qwen/Qwen3-0.6B-Base |
| Capas | 28 (todas full-attention) |
| d_model | 1024 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 (GQA) |
| head_dim | 64 |
| intermediate_size | 3072 |
| Vocabulario | 151,936 tokens |
| Parámetros | 616M |

### Cuantización

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Extra compression
)
```

**Uso de memoria**:
- VRAM: 514MB allocated, 600MB reserved
- RAM: ~1.5GB para tokenizer + overhead

---

## Corpus de Entrada

**Archivo**: `data/test_corpus.jsonl`

```
Num_texts:     50
Format:        JSONL (una línea por texto)
Max_length:    256 tokens (reducido para GTX 1650)
Batch_size:    1 (para evitar OOM)
```

### Muestra de textos

```json
{"text": "The transformer architecture revolutionized natural language processing..."}
{"text": "Optimal transport theory provides a framework for attention mechanisms..."}
...
```

---

## Configuración de Attention

### PlateauAttentionMechanism

```python
PlateauAttentionMechanism(
    d_model=1024,
    num_heads=16,
    epsilon=0.001,      # Sweet spot
    tau_iters=5,        # Sinkhorn iterations
    cost_type="l2_sq",  # L2 squared distance
)
```

### V4PlateauAdapter (NumPy)

```python
V4PlateauAdapter(
    d_model=1024,
    num_heads=16,
    num_experts=32,     # Expert-choice routing
    top_k=8,            # Top-k experts per token
    epsilon=0.001,
    use_fps_init=True,  # Farthest Point Sampling
)
```

---

## Parámetros del Sweep

### Epsilon Values

```python
epsilon_values = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
```

**Justificación**:
- ε → 0: Atención one-hot (máxima concentración, riesgo de colapso)
- ε = 0.001: Sweet spot encontrado
- ε → ∞: Atención uniforme (sin concentración)

### Target Layers

```python
target_layers = [0, 4, 8, 12, 16, 20, 24]  # 7 capas distribuidas
```

**Cobertura**:
- Layer 0: Embedding layer (vocabulary)
- Layers 4-8: Early processing
- Layers 12-16: Middle layers (bottleneck)
- Layers 20-24: Late layers (expressive representations)

---

## Métricas Calculadas

| Métrica | Descripción | Threshold |
|---------|-------------|-----------|
| effective_rank | Dimensiones efectivas via SVD | ≥ 50% baseline |
| concentration_ratio | Fracción activa de atención | Menor = mejor |
| anisotropy_index | Ratio eigenvalue max/suma | ≤ 0.5 |
| intrinsic_dim_mle | Dimensión del manifold (MLE) | ≥ 20 |
| attention_entropy | Entropía de la distribución | Menor = más peaked |

---

## Config Files Modificados

### experiments/config.py

```python
# Actualizado para Qwen3-0.6B
@dataclass
class AttentionConfig:
    d_model: int = 1024      # Era 2048 (Qwen 3.6)
    num_heads: int = 16
    head_dim: int = 64       # Era 128
    tau_iters: int = 5
    dropout: float = 0.0

@dataclass  
class Qwen3_06B_Config:
    model_name: str = "Qwen/Qwen3-0.6B-Base"
    num_layers: int = 28
    d_model: int = 1024
    num_attention_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 64
    target_layers: List[int] = [0, 4, 8, 12, 16, 20, 24]
```

### experiments/epsilon_sweep.py

```python
DEFAULT_D_MODEL = 1024  # Actualizado de 512
DEFAULT_NUM_HEADS = 16  # Actualizado de 8
```

---

## Seed y Reproducibilidad

```python
np.random.seed(42)
torch.manual_seed(42)
```

**Nota**: La cuantización 4-bit introduce variabilidad no determinista debido a la aproximación de las operaciones matriciales.

---

## Próximo Paso

→ Ver [02-extraccion-embeddings.md](./02-extraccion-embeddings.md) para el proceso de extracción.
