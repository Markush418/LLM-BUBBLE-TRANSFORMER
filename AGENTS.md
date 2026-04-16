# LLM-BUBBLE — Agent Instructions

**Project**: Bubble Transformer (Optimal Transport Attention)
**Core Question**: What ε maximizes embedding concentration without representational collapse?

---

## Commands

```bash
# Full experiment (mock mode, no GPU)
python experiments/run_experiment.py --mode mock

# Full experiment (real mode, needs GPU + Qwen3-0.6B)
python experiments/run_experiment.py --mode real

# Custom ε values
python experiments/run_experiment.py --epsilon-values 0.01 0.05 0.1

# Run tests
python -m pytest tests/ -v
python -m pytest tests/test_attention.py::TestPlateauAttentionMechanism -v

# Test individual modules (each has __main__ block)
python experiments/plateau_attention.py
python experiments/metrics.py
```

---

## Architecture

- **experiments/** — 13 Python modules, flat structure, sibling imports
- **NumPy contract**: `plateau_attention.py`, `metrics.py`, `epsilon_sweep.py` — pure NumPy only
- **PyTorch only for**: `extract_embeddings.py` (real model), `tests/` (test tensors)
- **Config**: Use `get_config()` from `experiments/config.py` — never hardcode values
- **Target layers**: [3, 7, 11, 15, 19, 23] (full-attention layers in Qwen 3.6)

---

## Critical Parameters

- **ε sweet spot**: ε → 0 = one-hot collapse; ε → ∞ = uniform; current optimal ≈ 0.001
- **Sinkhorn**: log-domain (`_sinkhorn_log_domain()`) prevents underflow at ε < 0.01
- **τ = 5 iterations**: Fixed convergence threshold
- **Seed = 42**: Reproducibility

---

## Anti-Patterns (CRITICAL)

- **DO NOT** use PyTorch tensors in `metrics.py` or `plateau_attention.py` — NumPy only
- **DO NOT** skip log-domain in Sinkhorn — numerical underflow
- **DO NOT** commit `venv312/`, `__pycache__/`, `*.npy`, `.ruff_cache/`
- **DO NOT** use bare `except:` — always specify exception type
- **DO NOT** suppress type errors with `as any` or `@ts-ignore`

---

## Known Issues

- Dual-head unpacking error in REAL mode: `expected 3 got 2` — PlateauAttention returns 2 values but callers expect 3. See `experiments/plateau_attention.py` line ~180 and `experiments/visualize.py` line ~61.
- Real mode requires GPU with ~8GB VRAM for Qwen3-0.6B
- Mock mode uses synthetic embeddings for testing without GPU