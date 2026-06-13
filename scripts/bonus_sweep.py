"""Sweep routing_bonus values to find sweet spot."""
import sys, os, time, math, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, r"C:\Users\negocio\Desktop\LLM-BUBBLE")

MODEL_ID = "Qwen/Qwen3-0.6B"
DTYPE = torch.float16
DEVICE = "cuda"
MAX_LENGTH = 256
STRIDE = 128

from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])[:5000]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def compute_ppl(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    total_nll = 0.0
    total_tokens = 0
    n_tokens = input_ids.shape[1]
    for i in range(0, n_tokens - MAX_LENGTH + 1, STRIDE):
        begin = i
        end = min(begin + MAX_LENGTH, n_tokens)
        input_chunk = input_ids[:, begin:end]
        with torch.no_grad():
            outputs = model(input_chunk)
            logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_chunk[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        num_tokens_in_chunk = shift_labels.numel()
        total_nll += loss.item() * num_tokens_in_chunk
        total_tokens += num_tokens_in_chunk
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_nll, 100))
    return ppl

# Baseline
print("Computing baseline PPL...")
model_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
ppl_base = compute_ppl(model_base, tokenizer, text)
print(f"  Baseline PPL: {ppl_base:.2f}")
del model_base
torch.cuda.empty_cache()

# Sweep
BONUS_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
results = {"baseline": ppl_base, "sweep": {}}

from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper

for bonus in BONUS_VALUES:
    print(f"\n--- bonus={bonus} ---")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)

    original_attn = model.model.layers[0].self_attn
    wrapper = Qwen3GQABubbleWrapper(
        original_attn=original_attn,
        num_bubbles=32,
        routing_bonus=bonus,
        debug=False,
    )
    model.model.layers[0].self_attn = wrapper

    ppl = compute_ppl(model, tokenizer, text)
    delta = ppl - ppl_base
    ratio = ppl / ppl_base
    print(f"  PPL: {ppl:.2f}  delta: {delta:+.2f}  ratio: {ratio:.4f}x")

    results["sweep"][str(bonus)] = {
        "ppl": ppl,
        "delta": delta,
        "ratio": ratio,
    }

    del model
    torch.cuda.empty_cache()

# Save
with open(r"C:\Users\negocio\Desktop\LLM-BUBBLE\bonus_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
print(f" SWEEP RESULTS")
print(f"{'=' * 60}")
print(f"  Baseline: {ppl_base:.2f}")
for bonus in BONUS_VALUES:
    r = results["sweep"][str(bonus)]
    print(f"  bonus={bonus}: PPL={r['ppl']:.2f}  delta={r['delta']:+.2f}  ratio={r['ratio']:.4f}x")
