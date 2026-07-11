"""Diagnose NaN gradient in learnable beta."""
import os, sys, time, torch, torch.nn.functional as F
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, 'experiments')

from transformers import AutoModelForCausalLM
from qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

tok_name = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(
    tok_name, torch_dtype=torch.float16, device_map='cuda',
    attn_implementation='eager',
)

layer = model.model.layers[12]
wrapper = Qwen3HybridGQABubbleWrapper(
    original_attn=layer.self_attn, epsilon=0.1, lam=1.0, use_delta=True,
    siri_mode='bias', bias_beta=0.24, learnable_beta=True, use_psi=False,
).cuda()
layer.self_attn = wrapper

# Freeze everything except beta
for name, p in model.named_parameters():
    if p is not wrapper.bias_beta:
        p.requires_grad = False

x = torch.randint(0, 1000, (1, 256), device='cuda')
with torch.enable_grad():
    out = model(x)
    logits = out.logits.float()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = x[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    print(f"loss = {loss.item():.4f}")
    print(f"loss requires_grad = {loss.requires_grad}")
    
    loss.backward()

bp = wrapper.bias_beta
print(f"beta = {bp.item():.4f}")
print(f"beta.grad = {bp.grad}")

# Check if gradient computation path is intact
# beta.grad should be: d(loss)/d(beta) = (d(loss)/d(out)) * out_siri
# If out_siri has NaN -> grad NaN
# If d(loss)/d(out) has NaN -> grad NaN
# If loss has NaN -> grad NaN
print(f"loss is nan: {torch.isnan(loss).item()}")
print(f"logits has nan: {torch.isnan(out.logits).any().item()}")
print(f"logits max: {out.logits.float().max().item():.2f}")
print(f"logits min: {out.logits.float().min().item():.2f}")

# Try with float32 to check if it's a half precision issue
print("\n--- Retrying with float32 ---")
model32 = AutoModelForCausalLM.from_pretrained(
    tok_name, torch_dtype=torch.float32, device_map='cuda',
    attn_implementation='eager',
)
layer32 = model32.model.layers[12]
wrapper32 = Qwen3HybridGQABubbleWrapper(
    original_attn=layer32.self_attn, epsilon=0.1, lam=1.0, use_delta=True,
    siri_mode='bias', bias_beta=0.24, learnable_beta=True, use_psi=False,
).cuda()
layer32.self_attn = wrapper32

for name, p in model32.named_parameters():
    if p is not wrapper32.bias_beta:
        p.requires_grad = False

with torch.enable_grad():
    out32 = model32(x)
    logits32 = out32.logits.float()
    shift_logits32 = logits32[:, :-1, :].contiguous()
    loss32 = F.cross_entropy(shift_logits32.view(-1, shift_logits32.size(-1)), shift_labels.view(-1))
    print(f"loss32 = {loss32.item():.4f}")
    loss32.backward()
    print(f"beta32.grad = {wrapper32.bias_beta.grad}")
