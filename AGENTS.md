# LLM-BUBBLE â€” Agent Instructions

**Project**: Bubble Transformer V4 (Hybrid DeltaNet + SIRI + Power Diagram)
**Core Question**: What Îµ maximizes embedding concentration without representational collapse?

---

## Commands

```bash
# Full experiment (mock mode, no GPU)
python experiments/run_experiment.py --mode mock

# Full experiment (real mode, needs GPU + Qwen3-0.6B)
python experiments/run_experiment.py --mode real

# Custom Îµ values
python experiments/run_experiment.py --epsilon-values 0.01 0.05 0.1

# Run tests
python -m pytest tests/ -v
python -m pytest tests/test_hybrid_attention.py::TestHybridAttention -v

# Test individual modules (each has __main__ block)
python experiments/plateau_attention.py        # SIRI log-domain Sinkhorn
python experiments/deltanet_attention.py       # DeltaNet delta rule
python experiments/power_diagrams.py           # Power Diagram psi
python experiments/siri_postprocess.py         # SIRI post-processing
python experiments/hybrid_attention.py         # Hybrid DeltaNet + SIRI + psi
```

---

## Architecture (post-SDOT, June 2026)

[DEFINITION] Hybrid Attention: DeltaNet (linear) + SIRI post-processing (doubly-stochastic) + Power Diagram psi (Laguerre bias).

```python
from hybrid_attention import HybridAttention

attn = HybridAttention(
    d_model=1024,        # Qwen3-0.6B
    num_heads=16,
    epsilon=0.1,         # SIRI bandwidth
    lam=0.5,             # 0=pure SIRI, 1=pure DeltaNet
    tau_iters=5,         # Sinkhorn iterations
)
out, attn = attn(x, return_attention=True)
# out: lam * out_delta + (1-lam) * out_siri
# attn: doubly-stochastic A_siri (if lam < 1)
```

- **experiments/** â€” 22 Python modules
  - **NumPy contract**: `plateau_attention.py`, `metrics.py`, `epsilon_sweep.py`, `deltanet_attention.py`, `power_diagrams.py`, `siri_postprocess.py`, `hybrid_attention.py` â€” pure NumPy
  - **PyTorch only for**: `extract_embeddings.py` (real model), `tests/` (test fixtures)
  - **Config**: Use `get_config()` from `experiments/config.py` â€” never hardcode values
  - **Target layers**: [3, 7, 11, 15, 19, 23] (full-attention layers in Qwen 3.6)
- **eps_sweet spot**: Îµ â‰ˆ 0.001 (preserved from original Bubble Transformer)
- **Hybrid lambda**: 0.5 default (balanced DeltaNet + SIRI)

---

## Critical Parameters

- **Îµ sweet spot**: Îµ â†’ 0 = one-hot collapse; Îµ â†’ âˆž = uniform; current optimal â‰ˆ 0.001
- **Sinkhorn**: log-domain (`_sinkhorn_log_domain()`) prevents underflow at Îµ < 0.01
- **Ï„ = 5 iterations**: Fixed convergence threshold
- **Î» (hybrid)**: 0.0 = pure SIRI, 0.5 = balanced, 1.0 = pure DeltaNet
- **Power Diagram Ïˆ**: learnable via `W_psi` projection (d_model â†’ 1)
- **Seed = 42**: Reproducibility across all modules

---

## Invariants (preserved across migrations)

These MUST be preserved in any architectural replacement:

- **I1**: C_ij = â€–Q_i - K_jâ€–Â² (geometric cost, NOT inner product)
- **I2**: A âˆˆ Î£_n (doubly-stochastic under SIRI)
- **I3**: log_S = -C/Îµ + Ïˆ (Power Diagram bias on log_Sinkhorn)
- **I4**: Îµ âˆˆ (0, âˆž), operational range [0.001, 1.0]
- **I5**: NumPy contract for core modules
- **I6**: Ï„ = 5 iterations (Sinkhorn convergence)

---

## Anti-Patterns (CRITICAL)

- **DO NOT** use PyTorch tensors in `metrics.py`, `plateau_attention.py`, `epsilon_sweep.py`, `deltanet_attention.py`, `power_diagrams.py`, `siri_postprocess.py`, `hybrid_attention.py` â€” NumPy only (defensive `_to_numpy()` accepts both)
- **DO NOT** skip log-domain in Sinkhorn â€” numerical underflow at Îµ < 0.01
- **DO NOT** modify `tensor_compat.py` without testing Python 3.14 compatibility
- **DO NOT** commit `venv312/`, `__pycache__/`, `*.npy`, `.ruff_cache/`
- **DO NOT** use bare `except:` â€” always specify exception type
- **DO NOT** suppress type errors with `as any` or `@ts-ignore`
- **DO NOT** use SDOT-specific APIs â€” SDOT has been removed (June 2026)
- **DO NOT** use `Mask[0,:]=0` in Sinkhorn â€” entire row masking breaks doubly-stochastic normalization

---

## Known Issues

- Real mode requires GPU with ~8GB VRAM for Qwen3-0.6B
- Mock mode uses synthetic embeddings for testing without GPU
- Dual-head attention (`DualHeadPlateauAttention`, legacy SIRI) returns 3 values `(output, A_low, A_high)` correctly. SDOT dual-head tests moved to legacy.
- Power Diagram Ïˆ as log_S bias is absorbed by Sinkhorn column normalization (mathematically expected). Effect: changes which keys get attention, not output magnitude. See test `test_power_diagram_psi_is_applied`.

---

## Recent Migration (June 2026)

- **Removed (June 2026)**: SDOT v3/v4 moved to `docs/legacy/sdot_v3_v4/`:
  - `models/sdot_attention.py`, `models/sdot_attention_v4.py`, `models/qwen3_gqa_bubble_wrapper.py`
  - `scripts/inject_sdot_qwen.py`
  - Associated tests: 8 SDOT-dependent test files (171 tests removed)
- **Added**: `deltanet_attention.py` (NeurIPS 2024), `siri_postprocess.py` (opt-in Sinkhorn), `power_diagrams.py` (psi as explicit module), `hybrid_attention.py` (combination)
- **Preserved**: SIRI (Sinkhorn log-domain), Power Diagram psi, BaroreceptorMLP, PlateauAttention (legacy SIRI)
- **Test status**: 393 passed, 21 skipped, 0 failed (post-cleanup; was 564 before SDOT removal)

See `docs/decisions/2026-06-27-sota-replacement-siri-preserved.md` for full migration rationale.



