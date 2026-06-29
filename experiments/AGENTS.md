# experiments/ — Bubble Transformer Source Code

**18 modules** — flat structure with `__init__.py`. All imports are sibling-relative.

---

## STRUCTURE

```
experiments/
├── run_experiment.py          # Main orchestrator — 4-step pipeline
├── config.py                  # Dataclass config + get_config()
├── plateau_attention.py       # SIRI: Sinkhorn-Knopp log-domain (NumPy) — KEEP
├── power_diagrams.py          # NEW (Jun 2026): Power Diagram psi as explicit module
├── deltanet_attention.py      # NEW (Jun 2026): DeltaNet delta rule (NeurIPS 2024)
├── siri_postprocess.py        # NEW (Jun 2026): SIRI as opt-in post-processing
├── hybrid_attention.py        # NEW (Jun 2026): DeltaNet + SIRI + psi combination
├── metrics.py                 # 6 concentration/geometry metrics (NumPy)
├── spectral_metrics.py        # SIGMA paper collapse detection
├── epsilon_sweep.py           # ε sweep controller + sweet spot finder
├── visualize.py               # 7 plot generators (matplotlib Agg, 200 DPI)
├── extract_embeddings.py      # Real Qwen model extraction (bfloat16, layer-by-layer)
├── generate_mock_embeddings.py # Synthetic embeddings (NumPy only)
├── tensor_compat.py           # NumPy fallback for PyTorch API (Python 3.14 compat)
├── analyze_results.py         # Post-hoc analysis script
└── v3_core.py, v4_adapter.py  # Legacy Bubble Transformer v3/v4 (kept for compat)
```

---

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Run experiment | `run_experiment.py` | `main()` — argparse CLI, 4-step pipeline |
| Config | `config.py` | `get_config()` returns `ExperimentConfig` dataclass |
| SIRI core | `plateau_attention.py` | `PlateauAttentionMechanism.forward()` — Sinkhorn log-domain |
| Power Diagram psi | `power_diagrams.py` | `PowerDiagramModule.compute_psi(K)` and `apply_to_log_sinkhorn(log_S, K)` |
| DeltaNet base | `deltanet_attention.py` | `DeltaNetAttention.forward()` — linear O(N) attention |
| SIRI post-process | `siri_postprocess.py` | `siri_sinkhorn_log_domain(log_S, tau)` and `siri_interpolate(out_sota, out_siri, lam)` |
| Hybrid | `hybrid_attention.py` | `HybridAttention.forward()` — DeltaNet + SIRI + psi |
| Metrics | `metrics.py` | `compute_all_metrics()` — 6 metrics in one call |
| Spectral | `spectral_metrics.py` | `compute_all_spectral_metrics()` — collapse detection |
| ε sweep | `epsilon_sweep.py` | `run_epsilon_sweep()` + `identify_sweet_spot()` |
| Plots | `visualize.py` | `generate_all_plots()` — 7 PNG outputs |
| Real embeddings | `extract_embeddings.py` | `QwenEmbeddingExtractor` — needs GPU + Qwen3-0.6B |
| Mock embeddings | `generate_mock_embeddings.py` | `save_mock_embeddings()` — NumPy synthetic |
| PyTorch compat | `tensor_compat.py` | `NumpyOps` singleton — torch API fallback |

---

## CONVENTIONS

- **NumPy contract**: `plateau_attention.py`, `metrics.py`, `epsilon_sweep.py`, `deltanet_attention.py`, `power_diagrams.py`, `siri_postprocess.py`, `hybrid_attention.py` — pure NumPy, `np.float32`
- **PyTorch only for**: `extract_embeddings.py` (real model extraction), `tests/` (test fixtures)
- **Defensive `_to_numpy()`**: All NumPy modules accept both NumPy arrays and torch tensors via internal conversion
- **Log-domain Sinkhorn**: `_sinkhorn_log_domain()` — prevents underflow at low ε
- **τ = 5 iterations**: Fixed Sinkhorn convergence threshold
- **Seed = 42**: Reproducibility across all modules
- **CLI pattern**: Each module has `if __name__ == "__main__"` — individually runnable
- **Import style**: `from module import func` — sibling imports without package prefix

---

## ANTI-PATTERNS

- **DO NOT** use PyTorch tensors in `metrics.py` or `plateau_attention.py` — NumPy only (defensive `_to_numpy()` accepts both)
- **DO NOT** skip log-domain in Sinkhorn — numerical underflow at ε < 0.01
- **DO NOT** modify `tensor_compat.py` without testing Python 3.14 compatibility
- **DO NOT** commit `__pycache__/` or `.pyc` files — already in `.gitignore`
- **DO NOT** use bare `except:` — always specify exception type

---

## NOTES

- **`optimal_config.py`**: Auto-generated from sweep analysis — not manually edited
- **`analyze_results.py`**: Post-hoc analysis — separate from main pipeline
- **`__init__.py`**: Present but minimal — modules still runnable independently via `sys.path`
- **SDOT removed (June 2026)**: `sdota_attention.py` and `sdot_attention_v4.py` are deprecated; use `hybrid_attention.py` instead
- **Hybrid attention (default)**: `HybridAttention(d_model, num_heads, epsilon, lam)` — combines DeltaNet + SIRI + psi with interpolation