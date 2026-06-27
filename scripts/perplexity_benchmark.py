"""
SIRI Perplexity Benchmark — Sprint A1
=====================================
Compara PPL de Qwen3-0.6B con:
  (A) softmax attention estándar (baseline)
  (B) Bubble Transformer en eps=EPS_STAR

Resultado esperado: dos números → PPL_baseline y PPL_bubble
Si PPL_bubble < PPL_baseline → SIRI tiene valor arquitectónico
Si PPL_bubble >= PPL_baseline → SIRI es geométrico pero sin ganancia downstream

Hardware mínimo: 4GB VRAM (GTX 1650 OK)
Tiempo estimado: ~15-25 min total en GTX 1650

Uso:
    python perplexity_benchmark.py

Requisitos:
    pip install torch transformers datasets tqdm
    # + las dependencias del repo LLM-BUBBLE (geoopt, etc.)
"""

import sys
import math
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ─── Configuración ────────────────────────────────────────────────────────────

MODEL_ID    = "Qwen/Qwen3-0.6B"
DATASET     = "wikitext"
DATASET_CFG = "wikitext-2-raw-v1"
SPLIT       = "test"

MAX_LENGTH  = 512    # ventana de contexto reducida
STRIDE      = 256    # overlap reducido (más rápido)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float16  # GTX 1650: float16 cabe, bfloat16 no soportado

EPS_STAR = 0.005 # SIRI regime (eps -> 0 = concentration without collapse)
SEED = 42
NUM_BUBBLES = 32
TOP_K = 64 # activates routing (NOT 1024 which disables it)
ROUTING_BONUS = 0.5 # additive bias for same-bubble pairs (0 = standard attn)

OUTPUT_FILE = Path("siri_ppl_results.json")

# ─── Reproducibilidad ─────────────────────────────────────────────────────────

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_wikitext2_text() -> str:
    """Descarga y concatena el split test de WikiText-2 (limitado para speed)."""
    print("-> Cargando WikiText-2 test split...")
    dataset = load_dataset(DATASET, DATASET_CFG, split=SPLIT)
    text = "\n\n".join(
        row["text"] for row in dataset if row["text"].strip()
    )
    # Limit to ~50k tokens for faster benchmark
    max_chars = 40000
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"  [LIMITED] {max_chars} chars")
    return text


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE,
    desc: str = "PPL",
) -> float:
    """
    Calcula perplexity con sliding window (estándar en literatura).
    PPL = exp( -1/N * sum log P(x_i | x_{<i}) )
    """
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    seq_len   = encodings.input_ids.size(1)

    print(f"  Tokens totales: {seq_len:,} | ventanas: {max_length} | stride: {stride}")

    nlls        = []
    total_toks  = 0
    prev_end    = 0

    pbar = tqdm(
        range(0, seq_len, stride),
        desc=f"  {desc}",
        unit="win",
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for begin in pbar:
            end      = min(begin + max_length, seq_len)
            trg_len  = end - prev_end
            inp_ids  = encodings.input_ids[:, begin:end].to(DEVICE)

            labels = inp_ids.clone()
            labels[:, :-trg_len] = -100

            out = model(inp_ids, labels=labels)
            nlls.append(out.loss.float() * trg_len)
            total_toks += trg_len
            prev_end   = end

            pbar.set_postfix({"running_ppl": f"{math.exp(sum(nlls).item() / total_toks):.2f}"})

            if end == seq_len:
                break

    ppl = math.exp(sum(nlls).item() / total_toks)
    return ppl


# ─── GQA-Native Bubble Wrapper ──────────────────────────────────────────────
#
# Uses Qwen3GQABubbleWrapper which preserves original Qwen3 projections.
# NO weight copy, NO reshape. Original q_proj/k_proj/v_proj/o_proj stay intact.
# Only replaces attention score computation with SDOT block-sparse + causal mask.


def _add_bubble_repo_to_path():
    repo_path = Path(r"C:\Users\negocio\Desktop\LLM-BUBBLE")
    if not repo_path.exists():
        raise FileNotFoundError(
            f"Repo no encontrado en {repo_path}\n"
            "Actualiza la ruta en _add_bubble_repo_to_path()"
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def swap_attention_layers(model, eps: float, target_layers: list = None) -> int:
    """
    Reemplaza self_attn de cada capa Qwen3 con Qwen3GQABubbleWrapper.

    NO copia pesos. NO hace reshape.
    Conserva todas las proyecciones originales (q_proj, k_proj, v_proj, o_proj).
    Usa q_norm/k_norm originales, position_embeddings desde DecoderLayer.
    """
    _add_bubble_repo_to_path()
    from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper

    n_swapped = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if target_layers is not None and layer_idx not in target_layers:
            continue

        original_attn = layer.self_attn

        print(f"  [Layer {layer_idx}] Wrapping with Qwen3GQABubbleWrapper"
              f" (eps={eps}, bubbles={NUM_BUBBLES}, bonus={ROUTING_BONUS})")

        wrapper = Qwen3GQABubbleWrapper(
            original_attn=original_attn,
            num_bubbles=NUM_BUBBLES,
            top_k=TOP_K,
            eps=eps,
            routing_bonus=ROUTING_BONUS,
            debug=(n_swapped == 0),
        )

        layer.self_attn = wrapper
        n_swapped += 1

    print(f"  [OK] {n_swapped} layers wrapped (no weight copy)")
    return n_swapped


# ─── Smoke Test ───────────────────────────────────────────────────────────────

def smoke_test_swap():
    """Verifica que el swap de atencion funciona en 1 capa antes del benchmark completo."""
    print("-- Smoke test: verificando swap en capa 0...")
    _add_bubble_repo_to_path()
    from models.qwen3_gqa_bubble_wrapper import Qwen3GQABubbleWrapper

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE
    )

    original_attn = model.model.layers[0].self_attn
    wrapper = Qwen3GQABubbleWrapper(
        original_attn=original_attn,
        num_bubbles=NUM_BUBBLES,
        top_k=TOP_K,
        eps=EPS_STAR,
        routing_bonus=ROUTING_BONUS,
        debug=True,
    )
    model.model.layers[0].self_attn = wrapper

    x = torch.randn(1, 10, model.config.hidden_size, device=DEVICE, dtype=DTYPE)
    pos = torch.arange(10).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = wrapper(x, position_ids=pos,
                      position_embeddings=model.model.rotary_emb(x, pos))[0]
    assert out.shape == (1, 10, model.config.hidden_size), f"Shape mismatch: {out.shape}"
    assert not out.isnan().any(), "NaN in output!"
    print(f"  [PASS] Smoke test PASSED: {x.shape} -> {out.shape}, no NaN")

    del model
    torch.cuda.empty_cache()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = {}

    print(f"\n{'='*60}")
    print(f" SIRI Perplexity Benchmark - WikiText-2")
    print(f" Device: {DEVICE} | dtype: {DTYPE}")
    print(f" eps*: {EPS_STAR} | bubbles: {NUM_BUBBLES} | top_k: {TOP_K} | bonus: {ROUTING_BONUS}")
    print(f" Wrapper: Qwen3GQABubbleWrapper (no weight copy)")
    print(f"{'='*60}\n")

    # Smoke test: verificar swap antes de proceder
    smoke_test_swap()

    text = load_wikitext2_text()

    # -- 1. Baseline: softmax estandar ------------------------------------
    print("-- [1/2] Baseline (softmax estandar) --")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=DEVICE,
    )
    model_base.eval()

    ppl_baseline = compute_perplexity(model_base, tokenizer, text, desc="Baseline")
    t_base = time.time() - t0

    print(f"\n OK PPL baseline = {ppl_baseline:.4f} ({t_base:.1f}s)\n")
    results["baseline"] = {"ppl": ppl_baseline, "time_s": t_base, "eps": "softmax"}

    del model_base
    torch.cuda.empty_cache()

    # -- 2. Bubble Transformer en eps* ------------------------------------
    print(f"-- [2/2] Bubble Transformer (eps={EPS_STAR}) --")
    t0 = time.time()

    model_bubble = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=DEVICE,
    )

    print(f" -> Swapeando attention layers (eps={EPS_STAR})...")
    try:
        n_swapped = swap_attention_layers(model_bubble, eps=EPS_STAR)
        print(f" OK {n_swapped} capas swapeadas")
    except Exception as e:
        print(f"\n X swap_attention_layers() falló:\n {e}")
        sys.exit(1)

    # Este código debe ejecutarse si el swap fue exitoso
    model_bubble.eval()

    ppl_bubble = compute_perplexity(model_bubble, tokenizer, text, desc=f"Bubble ε={EPS_STAR}")
    t_bubble = time.time() - t0

    print(f"\n  [PASS] PPL bubble = {ppl_bubble:.4f}  ({t_bubble:.1f}s)\n")
    results["bubble"] = {"ppl": ppl_bubble, "time_s": t_bubble, "eps": EPS_STAR}

    del model_bubble
    torch.cuda.empty_cache()

    # ── Reporte final ──────────────────────────────────────────────────────
    delta_ppl  = ppl_bubble - ppl_baseline
    delta_pct  = (delta_ppl / ppl_baseline) * 100
    verdict = "IMPROVEMENT" if delta_ppl < 0 else "REGRESSION" if delta_ppl > 0 else "EQUAL"

    results["summary"] = {
        "delta_ppl":    delta_ppl,
        "delta_pct":    delta_pct,
        "verdict":      verdict,
        "interpretation": (
            "SIRI tiene valor arquitectónico (PPL mejorada)"
            if delta_ppl < 0 else
            "SIRI es geométricamente real pero sin ganancia downstream — "
            "claims del paper deben limitarse a observación geométrica"
        ),
    }

    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"  PPL baseline (softmax):   {ppl_baseline:.4f}")
    print(f"  PPL bubble (eps={EPS_STAR}):  {ppl_bubble:.4f}")
    print(f"  Delta:                    {delta_ppl:+.4f}  ({delta_pct:+.2f}%)")
    print(f"  Veredicto:                {verdict}")
    print(f"{'='*60}\n")
    print(f"  -> Guardado en {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
