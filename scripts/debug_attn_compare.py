"""Debug: compare original Qwen3Attention output vs wrapper output on same input."""
import sys, os, torch
sys.path.insert(0, r"C:\Users\negocio\Desktop\LLM-BUBBLE")

from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"
DTYPE = torch.float16
DEVICE = "cuda"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)

# Get original attention and its output
original_attn = model.model.layers[0].self_attn
layer = model.model.layers[0]

# Create a test input
x = torch.randn(1, 16, 1024, dtype=DTYPE, device=DEVICE)
position_ids = torch.arange(16, device=DEVICE).unsqueeze(0)

# Compute position_embeddings the same way the model does
cos, sin = model.model.rotary_emb(x, position_ids)
position_embeddings = (cos, sin)

# Run through the full layer with original attention
layer.eval()
with torch.no_grad():
    # First: run the original layer
    residual = x.clone()
    normed = layer.input_layernorm(x)
    original_output, _ = original_attn(
        hidden_states=normed,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )
    original_residual_output = residual + original_output

print(f"Original attn output shape: {original_output.shape}")
print(f"Original output stats: mean={original_output.float().mean():.4f}, std={original_output.float().std():.4f}")
print(f"Original output has NaN: {original_output.isnan().any()}")

# Now swap and run through wrapper
from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper

wrapper = Qwen3GQABubbleWrapper(
    original_attn=original_attn,
    num_bubbles=32,
    routing_bonus=0.0,  # NO routing bias -- should match original
    debug=True,
)

with torch.no_grad():
    wrapper_output, _ = wrapper(
        hidden_states=normed,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )

print(f"\nWrapper output shape: {wrapper_output.shape}")
print(f"Wrapper output stats: mean={wrapper_output.float().mean():.4f}, std={wrapper_output.float().std():.4f}")
print(f"Wrapper output has NaN: {wrapper_output.isnan().any()}")

# Compare
diff = (original_output - wrapper_output).float()
print(f"\nDifference stats:")
print(f"  max abs diff: {diff.abs().max():.6f}")
print(f"  mean abs diff: {diff.abs().mean():.6f}")
print(f"  relative diff: {(diff.abs().mean() / original_output.float().abs().mean()):.6f}")

# Check if they're close
if diff.abs().max() < 0.01:
    print("  STATUS: MATCH (outputs are close)")
elif diff.abs().max() < 0.1:
    print("  STATUS: CLOSE (small differences)")
else:
    print("  STATUS: DIVERGENT (significant differences)")

    # Diagnose: check step by step
    print("\n--- Diagnosing step by step ---")
    Q = original_attn.q_proj(normed)
    K = original_attn.k_proj(normed)
    V = original_attn.v_proj(normed)
    print(f"Q proj output: mean={Q.float().mean():.6f}, std={Q.float().std():.6f}")
    print(f"K proj output: mean={K.float().mean():.6f}, std={K.float().std():.6f}")
    print(f"V proj output: mean={V.float().mean():.6f}, std={V.float().std():.6f}")
    print(f"Q proj weight shape: {original_attn.q_proj.weight.shape}")
    print(f"K proj weight shape: {original_attn.k_proj.weight.shape}")
    print(f"V proj weight shape: {original_attn.v_proj.weight.shape}")
    print(f"O proj weight shape: {original_attn.o_proj.weight.shape}")

    # Check what the original attention actually does
    import inspect
    src = inspect.getsource(type(original_attn).forward)
    print("\n--- Original Qwen3Attention.forward() source ---")
    print(src[:3000])
