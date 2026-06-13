"""Quick 1-layer PPL test: swap layer 0 only, measure perplexity on small sample."""
import sys, os, time, math, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, r"C:\Users\negocio\Desktop\LLM-BUBBLE")

MODEL_ID = "Qwen/Qwen3-0.6B"
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_BUBBLES = 32
TOP_K = 64
EPS_STAR = 0.005
ROUTING_BONUS = 0.1
MAX_LENGTH = 256
STRIDE = 128

def compute_ppl(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    total_nll = 0.0
    total_tokens = 0
    n_tokens = input_ids.shape[1]
    count = 0
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
        count += 1
        if count % 10 == 0:
            print(f"    chunk {count}, running avg NLL: {total_nll/total_tokens:.4f}")
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_nll, 100))
    return ppl

print("=" * 60)
print(" 1-Layer PPL Test: Qwen3GQABubbleWrapper")
print(f" Device: {DEVICE} | dtype: {DTYPE}")
print(f" eps: {EPS_STAR} | bubbles: {NUM_BUBBLES} | bonus: {ROUTING_BONUS}")
print(f" MAX_LENGTH: {MAX_LENGTH} | STRIDE: {STRIDE}")
print("=" * 60)

# Use a small sample from WikiText-2
print("\nLoading WikiText-2 test set (small sample)...")
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text_full = "\n\n".join(ds["text"])
# Take first 5000 chars to keep it fast on GTX 1650
text = text_full[:5000]
print(f"  Text length: {len(text)} chars")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# --- Baseline ---
print("\n[1/2] Baseline (standard softmax)...")
t0 = time.time()
model_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
ppl_base = compute_ppl(model_base, tokenizer, text)
t_base = time.time() - t0
print(f"  Baseline PPL: {ppl_base:.2f}  ({t_base:.1f}s)")
del model_base
torch.cuda.empty_cache()

# --- Bubble (1 layer) ---
print("\n[2/2] Bubble (layer 0 swapped)...")
t0 = time.time()
model_bubble = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)

from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper
original_attn = model_bubble.model.layers[0].self_attn

wrapper = Qwen3GQABubbleWrapper(
    original_attn=original_attn,
    num_bubbles=NUM_BUBBLES,
    top_k=TOP_K,
    eps=EPS_STAR,
    routing_bonus=ROUTING_BONUS,
    debug=True,
)
model_bubble.model.layers[0].self_attn = wrapper

ppl_bubble = compute_ppl(model_bubble, tokenizer, text)
t_bubble = time.time() - t0
print(f"  Bubble PPL: {ppl_bubble:.2f}  ({t_bubble:.1f}s)")
del model_bubble
torch.cuda.empty_cache()

# --- Results ---
print(f"\n{'=' * 60}")
print(f" RESULTS")
print(f"{'=' * 60}")
print(f"  Baseline PPL:  {ppl_base:.2f}")
print(f"  Bubble PPL:    {ppl_bubble:.2f}")
print(f"  Delta:          {ppl_bubble - ppl_base:+.2f}")
print(f"  Ratio:          {ppl_bubble / ppl_base:.4f}x")
if ppl_bubble < ppl_base:
    print(f"  STATUS:        IMPROVEMENT (bubble < baseline)")
elif ppl_bubble < ppl_base * 1.5:
    print(f"  STATUS:        ACCEPTABLE (within 50% of baseline)")
else:
    print(f"  STATUS:        DEGRADATION (too far from baseline)")
print(f"{'=' * 60}")

results = {
    "baseline_ppl": ppl_base,
    "bubble_ppl": ppl_bubble,
    "delta": ppl_bubble - ppl_base,
    "ratio": ppl_bubble / ppl_base,
    "config": {
        "model": MODEL_ID,
        "dtype": str(DTYPE),
        "device": DEVICE,
        "eps": EPS_STAR,
        "num_bubbles": NUM_BUBBLES,
        "top_k": TOP_K,
        "routing_bonus": ROUTING_BONUS,
        "layers_swapped": [0],
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "text_chars": len(text),
    }
}
out_path = r"C:\Users\negocio\Desktop\LLM-BUBBLE\siri_ppl_1layer_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to siri_ppl_1layer_results.json")
