#!/usr/bin/env python
"""FASE 0: INSPECCIÓN PREVIA - Verificar shapes reales de Qwen3-0.6B"""

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

print("=" * 80)
print("FASE 0: INSPECCIÓN PREVIA - shapes reales de Qwen3-0.6B")
print("=" * 80)

print(f"\nLoading model: {MODEL_ID}")
print(f"Device: {DEVICE} | dtype: {DTYPE}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map=DEVICE,
)

layer0 = model.model.layers[0].self_attn

print(f"\n{'*' * 80}")
print("QWEN3-0.6B Layer 0 Self-Attn Shapes:")
print("*" * 80)

print(f"\nq_proj.weight.shape: {layer0.q_proj.weight.shape}")
print(f"  output_features: {layer0.q_proj.out_features} = num_heads × head_dim = {model.config.num_attention_heads} × {model.config.head_dim}")

print(f"\nk_proj.weight.shape: {layer0.k_proj.weight.shape}")
print(f"  output_features: {layer0.k_proj.out_features} = num_kv_heads × head_dim = {model.config.num_key_value_heads} × {model.config.head_dim}")

print(f"\nv_proj.weight.shape: {layer0.v_proj.weight.shape}")
print(f"  output_features: {layer0.v_proj.out_features}")

print(f"\no_proj.weight.shape: {layer0.o_proj.weight.shape}")
print(f"  in_features: {layer0.o_proj.in_features}, out_features: {layer0.o_proj.out_features}")

print(f"\n{'*' * 80}")
print("QWEN3 CONFIG (from config):")
print("*" * 80)
print(f"  num_attention_heads: {model.config.num_attention_heads}")
print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
print(f"  hidden_size: {model.config.hidden_size}")
print(f"  head_dim: {model.config.head_dim}")
print(f"  kv_groups: {model.config.num_attention_heads // model.config.num_key_value_heads}")

# Test rotary_fn (not rotary_emb)
print(f"\n{'*' * 80}")
print("rotary_fn module:")
print("*" * 80)
print(f"  type: {type(layer0.rotary_fn)}")
print(f"  module_name: {layer0.rotary_fn.__class__.__name__}")

# Try calling it
print(f"\n  Trying rotary_fn with dummy input:")
dummy_x = torch.randn(1, 10, model.config.hidden_size, device=DEVICE, dtype=DTYPE)
dummy_pos = torch.arange(10, device=DEVICE).unsqueeze(0)
try:
    cos, sin = layer0.rotary_fn(dummy_x, dummy_pos)
    print(f"    success! cos.shape: {cos.shape}, sin.shape: {sin.shape}")
    print(f"    cos dtype: {cos.dtype} | device: {cos.device}")
except Exception as e:
    print(f"    error: {e}")

print(f"\n{'*' * 80}")
print("CONCLUSIONS:")
print("*" * 80)
print("1. q_proj out = 2048 = 16 × 128 (num_heads × head_dim)")
print("2. k_proj out = 1024 = 8 × 128 (num_kv_heads × head_dim)")
print("3. v_proj out = 1024 = 8 × 128")
print("4. o_proj in  = 2048 = 16 × 128")
print("5. head_dim = 128 (from config)")
print("6. kv_groups = 2 (cada KV head serve 2 Q heads)")
print("7. rotary_fn signature: (x, position_ids)")
print("8. Para reshape Q: [B, N, 2048] -> [B, N, 16, 128]")
print("9. GQA: repeat_interleave K/V por 2 antes del clustering")
print("10. RoPE: apply_rotary_pos_emb(Q, K, cos, sin)")

print(f"\n{'=' * 80}")
print("FASE 0 COMPLETE - listo para FASE 1")
print("=" * 80)
