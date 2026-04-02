# LLM-BUBBLE — PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-02
**Project Type:** Python ML Research — Bubble Transformer (Optimal Transport Attention)
**Core Question:** What ε (viscosity) maximizes embedding concentration without representational collapse?

---

## STRUCTURE

```
LLM-BUBBLE/
├── experiments/          # All source code — 13 modules + __init__.py
├── data/                 # Input corpus (test_corpus.jsonl)
├── embeddings/           # Generated .npy files (softmax + plateau)
├── results/              # Sweep JSON + sweet spot MD report
├── plots/                # 7 PNG visualizations (200 DPI, headless Agg)
├── docs/
│   ├── legacy/           # Historical docs: texto.txt, pyth.txt, readme.rtf, 2ABRIL.txt
│   └── superpowers/specs/  # Design specification
├── tests/                # Single unittest file (28 tests)
├── venv312/              # Python 3.12 venv — DO NOT commit
├── requirements.txt      # 11 deps: torch, transformers, scipy, matplotlib, seaborn, sklearn, umap-learn
├── .gitignore            # Excludes venv, *.npy, __pycache__, plots/, results/
└── LICENSE               # MIT License
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Run full experiment | `experiments/run_experiment.py` | `--mode mock` (no GPU) or `--mode real` (Qwen3-0.6B 4-bit) |
| Attention mechanism | `experiments/plateau_attention.py` | Sinkhorn-Knopp log-domain, multi-head |
| Metrics engine | `experiments/metrics.py` | 6 metrics: eff_rank, intrinsic_dim, anisotropy, pairwise_dist, concentration, entropy |
| ε sweep logic | `experiments/epsilon_sweep.py` | Iterates ε × layers, computes all metrics |
| Configuration | `experiments/config.py` | Dataclass singleton `get_config()` |
| Visualizations | `experiments/visualize.py` | 7 plot types: rank curves, heatmaps, Pareto, t-SNE, anisotropy, intrinsic_dim, dashboard |
| Test suite | `tests/test_pipeline.py` | `python tests/test_pipeline.py` — unittest, mock-based |
| Embedding extraction | `experiments/extract_embeddings.py` | Real Qwen model, bfloat16, layer-by-layer |
| Mock embeddings | `experiments/generate_mock_embeddings.py` | Synthetic, NumPy only |
| Results | `results/epsilon_sweep.json` | 1068 lines, 9 ε × 6 layers |
| Sweet spot report | `results/sweet_spot_analysis.md` | Auto-generated recommendation |

## CONVENTIONS

- **NumPy-first**: Core modules use `np.float32`, not PyTorch tensors. Tests bridge both.
- **Log-domain Sinkhorn**: Prevents underflow. `log_S = -C / ε`, then `u, v` dual potentials.
- **τ = 5 iterations**: Sinkhorn convergence threshold (verified in Sinkformers paper).
- **Random seed = 42**: Fixed for reproducibility across all modules.
- **Headless rendering**: `matplotlib.use("Agg")` — no GUI display needed.
- **Mode auto-detection**: `embeddings/metadata.json` determines mock vs real at runtime.
- **CLI overrides**: All config defaults overridable via argparse in `run_experiment.py`.

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT commit**: `venv312/`, `__pycache__/`, `.ruff_cache/`, `*.npy`
- **DO NOT use** PyTorch in core pipeline — NumPy only (PyTorch only for real model extraction)
- **DO NOT skip** log-domain in Sinkhorn — numerical underflow at low ε
- **NEVER** use `torch` tensors in `metrics.py` or `plateau_attention.py` — pure NumPy contract

## UNIQUE STYLES

- **Dual-mode architecture**: Mock (NumPy synthetic) and Real (Qwen3-0.6B 4-bit NF4 quantized) share the same pipeline
- **tensor_compat.py**: NumPy fallback for PyTorch API — Python 3.14 Windows compatibility layer
- **Embeddings stored in-repo**: `.npy` files in `embeddings/softmax/` (24 layers) — unusual for research projects

## COMMANDS

```bash
# Full experiment (mock mode, no GPU)
python experiments/run_experiment.py --mode mock

# Full experiment (real mode, needs GPU + Qwen3-0.6B)
python experiments/run_experiment.py --mode real

# Custom ε values
python experiments/run_experiment.py --epsilon-values 0.01 0.05 0.1 0.5

# Custom target layers
python experiments/run_experiment.py --target-layers 3 7 11 15

# Skip embedding generation (use existing)
python experiments/run_experiment.py --skip-generation

# Run tests
python tests/test_pipeline.py

# Install deps
pip install -r requirements.txt
```

## NOTES

- **Qwen 3.6 attention**: Hybrid architecture — 3 layers DeltaNet + 1 full attention. Target layers: [3, 7, 11, 15, 19, 23]
- **ε sweet spot**: ε → 0 = one-hot collapse; ε → ∞ = uniform. Target: max concentration with effective_rank ≥ 50% baseline
- **Current results**: ε=0.001 optimal (from `results/sweet_spot_analysis.md`)
- **embeddings/plateau/** is EMPTY — PlateauAttention embeddings not yet generated
- **No `pyproject.toml`** — plain requirements.txt project, no packaging
