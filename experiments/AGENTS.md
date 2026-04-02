# experiments/ — Bubble Transformer Source Code

**13 modules** — flat structure with `__init__.py`. All imports are sibling-relative.

---

## STRUCTURE

```
experiments/
├── run_experiment.py          # Main orchestrator — 4-step pipeline
├── config.py                  # Dataclass config + get_config() singleton
├── plateau_attention.py       # Sinkhorn-Knopp log-domain attention (NumPy)
├── metrics.py                 # 6 concentration/geometry metrics (NumPy)
├── epsilon_sweep.py           # ε sweep controller + sweet spot finder
├── visualize.py               # 7 plot generators (matplotlib Agg, 200 DPI)
├── extract_embeddings.py      # Real Qwen model extraction (bfloat16, layer-by-layer)
├── generate_mock_embeddings.py # Synthetic embeddings (NumPy only)
├── tensor_compat.py           # NumPy fallback for PyTorch API (Python 3.14 compat)
├── analyze_results.py         # Post-hoc analysis script
└── optimal_config.py          # Auto-generated: ε=0.001, layers [3,7,11]
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Run experiment | `run_experiment.py` | `main()` — argparse CLI, 4-step pipeline |
| Config | `config.py` | `get_config()` returns `ExperimentConfig` dataclass |
| Attention core | `plateau_attention.py` | `PlateauAttentionMechanism.forward()` — Sinkhorn log-domain |
| Metrics | `metrics.py` | `compute_all_metrics()` — 6 metrics in one call |
| ε sweep | `epsilon_sweep.py` | `run_epsilon_sweep()` + `identify_sweet_spot()` |
| Plots | `visualize.py` | `generate_all_plots()` — 7 PNG outputs |
| Real embeddings | `extract_embeddings.py` | `QwenEmbeddingExtractor` — needs GPU + Qwen3-0.6B |
| Mock embeddings | `generate_mock_embeddings.py` | `save_mock_embeddings()` — NumPy synthetic |
| PyTorch compat | `tensor_compat.py` | `NumpyOps` singleton — torch API fallback |

## CONVENTIONS

- **NumPy contract**: `plateau_attention.py`, `metrics.py`, `epsilon_sweep.py` — pure NumPy, `np.float32`
- **PyTorch only for**: `extract_embeddings.py` (real model extraction), `tests/test_pipeline.py` (test tensors)
- **Log-domain Sinkhorn**: `_sinkhorn_log_domain()` — prevents underflow at low ε
- **τ = 5 iterations**: Fixed Sinkhorn convergence threshold
- **Seed = 42**: Reproducibility across all modules
- **CLI pattern**: Each module has `if __name__ == "__main__"` — individually runnable
- **Import style**: `from module import func` — sibling imports without package prefix

## ANTI-PATTERNS

- **DO NOT** use PyTorch tensors in `metrics.py` or `plateau_attention.py` — NumPy only
- **DO NOT** skip log-domain in Sinkhorn — numerical underflow at ε < 0.01
- **DO NOT** modify `tensor_compat.py` without testing Python 3.14 compatibility
- **DO NOT** commit `__pycache__/` or `.pyc` files — already in `.gitignore`

## NOTES

- **`optimal_config.py`**: Auto-generated from sweep analysis — not manually edited
- **`analyze_results.py`**: Post-hoc analysis — separate from main pipeline
- **`__init__.py`**: Present but minimal — modules still runnable independently via `sys.path`
