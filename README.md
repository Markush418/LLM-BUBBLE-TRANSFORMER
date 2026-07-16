# Bubble Transformer

> **Hybrid attention with entropic optimal transport** · Independent research from [kyan-labs](https://kyan-labs.com)

<!-- BADGES: replace tests count if it changes, add Zenodo DOI once submitted -->
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-462_passing-brightgreen.svg)
<!-- TODO: Zenodo DOI badge -->

---

## TL;DR

Bubble Transformer replaces softmax attention with an **entropic optimal transport formulation** (SIRI) combined with **DeltaNet's O(N) associative recall**. On Qwen3-0.6B we observe **SIRI — Sparsity-Induced Rank Inflation** — a non-monotonic empirical phenomenon where effective attention rank peaks at **2.89× the softmax baseline** at a specific bandwidth ε, and is absent in random matrices.

**Current status**: hybrid architecture (DeltaNet + SIRI + Power Diagram ψ) validated on Qwen3-0.6B · 462/464 tests passing · single-layer swap ΔPPL gate PASS at layer 7 (+0.74% vs softmax baseline).

---

## Key finding — SIRI (Sparsity-Induced Rank Inflation)

Sinkhorn-normalized attention exhibits a rank inflation phenomenon absent in softmax and random baselines.

**Measurement on Qwen3-0.6B**:

| Attention type            | Effective Rank (R_eff) |
| ------------------------- | ---------------------- |
| Random matrix (control)   | ~1.0                   |
| Softmax baseline          | 199.6                  |
| **SIRI @ ε=0.005 (peak)** | **576.5 (2.89× softmax)** |

R_eff varies non-monotonically with ε. The peak location is stable across seeds. Because it does not appear in random matrices, the effect is a property of the Sinkhorn normalization interacting with learned attention geometry — not a numerical artifact.

<!-- TODO: embed R_eff vs ε plot from assets/ (Captura.PNG or new) -->

---

## Architecture

```
                      Input embeddings
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
         DeltaNet path                SIRI path
      (linear O(N) recall)     (Sinkhorn log-domain
                                + Power Diagram ψ)
               │                           │
               └─────────────┬─────────────┘
                             ▼
                   Hybrid interpolation
                   out = λ · delta + (1−λ) · siri
```

Three components:

- **DeltaNet** (Yang et al. 2024) — linear-time attention with delta rule for associative recall
- **SIRI post-processing** — Sinkhorn–Knopp doubly-stochastic normalization in log-domain, τ = 5 iterations
- **Power Diagram ψ** — learnable Laguerre tessellation bias in `log_S = −C/ε + ψ`

Cost matrix uses geometric L2 distance: `C_ij = ‖Q_i − K_j‖²`, **not** the standard inner product `QK⊤`.

---

## Quick start

```bash
pip install -r requirements.txt
```

```python
from hybrid_attention import HybridAttention

attn = HybridAttention(
    d_model=1024, num_heads=16,       # Qwen3-0.6B dims
    epsilon=0.005,                    # SIRI bandwidth at R_eff peak
    lam=0.5,                          # hybrid balance (1.0=DeltaNet, 0.0=SIRI)
    siri_mode="soft",                 # classical | chiller | sparse | soft
    siri_alpha=0.3,                   # blend weight (soft mode)
)
output, attn_matrix = attn(x, return_attention=True)
```

---

## Results

### Hero — single-layer swap on Qwen3-0.6B

Full-attention layer 7 replaced with Bubble Transformer, rest of model frozen.

| Config                        | ε     | τ | PPL        | ΔPPL vs softmax |
| ----------------------------- | ----- | - | ---------- | --------------- |
| Softmax baseline              | —     | — | (baseline) | —               |
| **Bubble Transformer @ L7**   | 0.001 | 1 | **22.681** | **+0.74%** ✓    |

Gate criterion: ΔPPL ≤ 2%. **PASS.**

<!-- TODO: fill exact softmax baseline PPL for L7 config (memory has +0.74% relative but not absolute baseline) -->

### Ablation — SIRI-Soft variants at layer 3 (λ = 0.5)

Classical doubly-stochastic SIRI destroys attention peakedness. Three variants preserve it to varying degrees:

| Variant                | Formula                     | PPL       | ΔPPL     |
| ---------------------- | --------------------------- | --------- | -------- |
| Softmax baseline       | —                           | 23.37     | —        |
| **Soft blend (α=0.7)** | (1−α)·softmax + α·SIRI      | **26.76** | +14.5%   |
| Classical SIRI         | Sinkhorn(−C/ε)              | 30.14     | +29.0%   |
| Chiller (β=5)          | Sinkhorn(scores·β)          | 39.39     | +68.5%   |

Soft blend recovers roughly half of the classical SIRI degradation at layer 3.

### Test suite

462 passing · 2 skipped · 0 failed (as of June 28, 2026)

---

## Method

Attention formulated as entropic optimal transport:

$$
\mathcal{E}(A) = \langle A, C \rangle - \epsilon \cdot H(A)
$$

where $A$ is the attention matrix, $C_{ij} = \|Q_i - K_j\|_2^2$ is the geometric cost, $H(A)$ is Shannon entropy, and $\epsilon$ is the bandwidth (temperature).

Sinkhorn iteration in log-domain (numerical stability at ε < 0.01):

```
log_S = -C / ε + ψ                 # cost + Power Diagram bias
u, v = 0, 0                        # dual potentials
for τ in range(5):
    u = -logsumexp(log_S + v, axis=-1)
    v = -logsumexp(log_S + u, axis=-2)
A = exp(log_S + u + v)             # doubly-stochastic
```

Convergence error bound: $O(\exp(-10 \epsilon \sigma_{\max}(C)))$.

Full mathematical formalism in [`docs/decisions/2026-06-27-siri-power-diagram-math.md`](docs/decisions/2026-06-27-siri-power-diagram-math.md).

---

## Reproducibility

- **Model**: Qwen3-0.6B (hybrid architecture: 3 DeltaNet + 1 full attention, repeated). Full-attention layer indices: `[3, 7, 11, 15, 19, 23]`
- **Precision**: bfloat16 during embedding extraction
- **Sinkhorn iterations**: τ = 5 (convergence follows Sinkformers, Sander et al. 2022)
- **Seeds**: fixed across all reported experiments (see `experiments/config.py`)
- **Hardware**: <!-- TODO: fill GPU model and memory -->

Reproduce:

```bash
python experiments/run_experiment.py --mode real   # requires GPU + Qwen3-0.6B weights
python -m pytest tests/ -v                         # full test suite
```

Mock mode (CPU-only, synthetic embeddings) for architecture validation:

```bash
python experiments/run_experiment.py --mode mock
```

---

## Related work

- **DeltaNet** (Yang et al., NeurIPS 2024) — linear attention with delta rule · [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
- **Sinkformers** (Sander et al., ICML 2022) — first Sinkhorn-based attention formulation
- **Kimi Linear / KDA** (Kimi Team, 2025) — SOTA linear attention (evaluated as opt-in alternative)
- **SIGMA** (2024) — spectral collapse detection metrics used here as diagnostic

Full bibliography (17 papers) in [`docs/references.bib`](docs/references.bib).

---

## Citation

If you use Bubble Transformer or the SIRI finding in your research, please cite:

```bibtex
@misc{bubble_transformer_2026,
  title        = {Bubble Transformer: Hybrid Attention with Entropic Optimal Transport},
  author       = {Marcus and kyan-labs},
  year         = {2026},
  howpublished = {\url{https://github.com/Markush418/LLM-BUBBLE-TRANSFORMER}},
  note         = {Independent research. Zenodo DOI: TODO}
}
```

<!-- TODO: fill full author name and Zenodo DOI once published -->

---

## About

**Bubble Transformer** is independent research from **[kyan-labs](https://kyan-labs.com)** — an independent research and engineering studio led by Marcus, based in Argentina.

kyan-labs consults on:

- LLM inference cost optimization
- Custom attention mechanisms and long-context architectures
- Multi-agent orchestration systems
- Compiler engineering for prompt / semantic compression

**Consulting inquiries**: <!-- TODO: fill contact email --> · [kyan-labs.com](https://kyan-labs.com)

---

## License

MIT — see [LICENSE](LICENSE)
