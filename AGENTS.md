# LLM-BUBBLE — Agent Instructions

**Project**: Bubble Transformer V5 (Focus-Inspired Sinkhorn Token Grouping)
**Status**: Research artifact — single-layer swap validated on Qwen3-0.6B, gate PASSED at +0.16% PPL.
**Core question (V5)**: How do we preserve softmax peakedness while injecting Sinkhorn-based geometric structure into causal LM attention?

The V4 question ("what ε maximizes concentration without collapse?") has been reframed. The gate is **ΔPPL ≤ 2%** on Qwen3-0.6B — every other metric is diagnostic, not decisive. See `IMPORTANTE/BT-V5_05_protocol_positioning.md`.

---

## Commands

```bash
# Gate-passing reproduction (~30 min on GTX 1650, 4.3 GB VRAM)
python experiments/benchmark_focus_fine_sweep.py             # ε, τ sweep at L12
python experiments/benchmark_focus_layer_sweep_optimal.py    # layer sweep at optimum
python experiments/benchmark_focus_deltanet_sweep.py         # λ sweep at L7 (best config)
python experiments/benchmark_focus_multilayer.py             # multi-layer scaling
python experiments/niah_focus_bubble.py                      # Needle-in-a-Haystack at 2K

# Legacy V4 mock/real experiments (still work, not the current story)
python experiments/run_experiment.py --mode mock
python experiments/run_experiment.py --mode real

# Tests
python -m pytest tests/ -v                                   # 490 collected; 468 pass in a bare env, ~489 on the dev box (14 wrapper errors + 1 CUDA test need cached Qwen3 + GPU)
python -m pytest tests/test_focus_bubble_attention.py -v     # V5 core
python -m pytest tests/test_hybrid_attention.py -v           # V4 hybrid (kept)

# Standalone module smoke tests (each has __main__)
python experiments/focus_bubble_attention.py
python experiments/deltanet_attention.py
python experiments/siri_postprocess.py
python experiments/hybrid_attention.py
python experiments/plateau_attention.py     # legacy SIRI, kept
```

Every benchmark writes to `results_real/focus_bubble/*.json`. Compare against numbers in the README table before declaring success.

---

## Architecture (V5, July 2026)

[DEFINITION] **Focus Bubble** = Sinkhorn as a token-grouping operator, softmax as the attention operator. This is a paradigm shift from V1–V4, which used Sinkhorn to replace softmax and paid +30% to +211% PPL for it.

The V5 core forward pass (`experiments/focus_bubble_attention.py`):

```python
S      = Q @ K.T / sqrt(d) + psi           # dot-product scores + Power Diagram bias
S     += causal_mask
groups = sinkhorn_log(S * epsilon, tau=1)   # doubly-stochastic soft grouping
attn   = softmax(S + log(groups + 1e-10))   # softmax within groups → peaked
out    = attn @ V
```

Combined variant `FocusBubbleDeltaNet` interpolates with a DeltaNet linear-attention branch:

```python
out = lam * out_delta + (1 - lam) * out_focus_bubble
# best empirical config: lam = 0.3 at layer 7 → +0.16% PPL vs softmax baseline
```

### Optimal single-layer configuration (measured)
- **Layer**: 7 (out of 28 in Qwen3-0.6B)
- **ε (epsilon)**: 0.001
- **τ (Sinkhorn iterations)**: 1
- **λ (DeltaNet blend)**: 0.3

### Multi-layer scaling (measured)
- FocusDeltaNet at {L7, L10, L12}: +1.38% PPL — still inside the gate.
- Pure Focus (without DeltaNet blend) degrades past 2 simultaneous layers.

### Long-context (measured, limited)
- Needle-in-a-Haystack at 2K tokens: 100% retrieval on every tested configuration.
- 4K–32K NIAH and RULER are unmeasured — do not claim otherwise.

---

## Codebase map

- **`experiments/`** — 60+ modules. The ones that matter now:
  - `focus_bubble_attention.py` — V5 core (PyTorch)
  - `qwen3_focus_bubble_wrapper.py` — drop-in Qwen3 attention swap
  - `benchmark_focus_*.py` — reproduction scripts
  - `niah_focus_bubble.py` — long-context sanity
  - `deltanet_attention.py` — DeltaNet (Yang et al. 2024, NumPy)
  - `siri_postprocess.py`, `siri_soft.py` — Sinkhorn post-processing + soft variants (NumPy)
  - `plateau_attention.py` — legacy SIRI attention (NumPy, kept for reference)
  - `hybrid_attention.py` — V4 DeltaNet + SIRI + ψ combinator (NumPy)
  - `power_diagrams.py` — Laguerre bias ψ (kept, but see honest caveat below)
  - `metrics.py`, `spectral_metrics.py`, `epsilon_sweep.py` — diagnostics (NumPy)
  - `config.py` — `get_config()`; never hardcode
- **`models/`** — PyTorch wrappers.
- **`tests/`** — 30 test files, 490 tests collected. Bare-env baseline: 468 passing / 1 skipped / 7 failed (6 NumPy 2.4 numerical + 1 CUDA-absent) / 14 errored (missing Qwen3 cache). On the dev box with GPU + cached Qwen3: 14 wrapper errors and 1 CUDA test flip to pass.
- **`paper/main.tex`** — arXiv preprint (V5 Focus Bubble).
- **`docs/blog_post.md`** — public writeup of the gate-passing result.
- **`IMPORTANTE/BT-V5_*.md`** — 7 formal papers (0=bases, I=arch, II=SIRI, III=diff/routing, IV=complexity, V=protocol, VI=focus). These are the source of truth for anything design-level.
- **`docs/decisions/`** — architectural decision records (SDOT removal, SIRI preservation).
- **`docs/legacy/sdot_v3_v4/`** — archived SDOT code + tests (removed June 2026).
- **`snap/`** — LLM-SNAP integration (adjacent project).

---

## Critical parameters

| Parameter | Value | Notes |
|---|---|---|
| ε (SIRI bandwidth) | **0.001** | Sweet spot for L7. Log-domain protects against underflow at ε<0.01. |
| τ (Sinkhorn iterations) | **1** for Focus Bubble; **5** for legacy SIRI | Focus needs only 1 because softmax handles the sharpness. |
| λ (DeltaNet blend) | **0.3** | 30% linear delta rule, 70% Focus grouping. |
| Target layer (single-swap) | **7** | 8 of 10 tested layers pass the gate; L9 is an unresolved anomaly. |
| Gate threshold | ΔPPL ≤ 2% | The absolute arbiter — every other metric is diagnostic. |
| Seed | 42 | Fixed across all modules. |

---

## Invariants (I1–I6)

Preserve these across any architectural change. If you need to break one, document it explicitly (see the V5 caveats below for how it's done).

- **I1** — Geometric cost: $C_{ij} = \|Q_i - K_j\|_2^2$ (not inner product) in the OT formulation.
- **I2** — Doubly-stochastic under SIRI: $A \in \Sigma_n$ (Birkhoff polytope).
- **I3** — Power Diagram bias: $\log S = -C/\varepsilon + \psi$.
- **I4** — ε ∈ (0, ∞), operational range [0.001, 1.0].
- **I5** — **NumPy contract** for pure-math modules: no PyTorch in `plateau_attention.py`, `metrics.py`, `epsilon_sweep.py`, `deltanet_attention.py`, `power_diagrams.py`, `siri_postprocess.py`, `hybrid_attention.py`. Defensive `_to_numpy()` accepts both.
- **I6** — τ = 5 iterations for legacy SIRI Sinkhorn convergence. Focus Bubble uses τ = 1 (justified: softmax within groups handles peakedness).

**V5 exceptions (documented, not accidental)**:
- Focus Bubble uses **dot-product scores** ($QK^\top/\sqrt{d}$), not $\|Q-K\|^2$ — violates I1 intentionally. This is what preserves peakedness.
- Sinkhorn output in Focus Bubble is a **grouping mask**, not the attention itself — I2 still holds for `groups`, but the actual attention `softmax(S + log(groups))` is not doubly-stochastic.

---

## Anti-patterns (CRITICAL — will break something)

- **DO NOT** use PyTorch tensors in the NumPy-contract modules (I5 list above).
- **DO NOT** skip log-domain in Sinkhorn — silent underflow at ε<0.01.
- **DO NOT** normalize an entire row to zero (`Mask[0, :] = 0`) in Sinkhorn — breaks doubly-stochastic invariance.
- **DO NOT** replace softmax with a doubly-stochastic distribution in a causal LM. V1–V4 tried this six ways; every attempt failed the gate. This is the pit V5 climbed out of.
- **DO NOT** use SDOT APIs — SDOT v3/v4 was removed June 2026, archived under `docs/legacy/sdot_v3_v4/`.
- **DO NOT** import from removed modules: `models/sdot_attention.py`, `models/sdot_attention_v4.py`, `models/qwen3_gqa_bubble_wrapper.py`, `scripts/inject_sdot_qwen.py`.
- **DO NOT** modify `tensor_compat.py` without testing Python 3.14 compatibility.
- **DO NOT** commit `venv312/`, `__pycache__/`, `*.npy`, `.ruff_cache/`.
- **DO NOT** use bare `except:` — always specify exception type.
- **DO NOT** claim wall-clock speedup, downstream benchmark results, or long-context (>2K) retrieval — none of these are measured. See "Not claimed" in the README.
- **DO NOT** cite the one-sided EOT foundation as "Daneshmand" — the correct attribution is **Litman, E. (2025), arXiv:2508.08369**. Daneshmand's paper (arXiv:2410.19931) is a different, adjacent result. See `IMPORTANTE/BT-V5_05_protocol_positioning.md` §4.
- **DO NOT** apply RoPE before Q/K projection or after clustering — the correct order is `Q,K = proj(x); Q,K = rotate(Q,K); route(...)`. Wrong order caused the PPL=831,974 incident.

---

## Honest caveats (say the quiet part out loud)

- **Power Diagram ψ is absorbed** by Sinkhorn column normalization in Focus Bubble — its empirical effect is zero (verified: L12 FocusOnly ψ=on/off gives identical PPL=23.082). It stays in the code for research continuity, not because it currently contributes.
- **L9 fails the gate** (+2.28%) while L10/L11/L12 pass. Per-head analysis shows outlier heads (H2: 131% L2 ratio, H6: 154%), but neighboring layers with similar outliers pass. Root cause: unknown, likely a per-head integration effect.
- **All results are single-layer swaps on frozen Qwen3-0.6B.** No pretrain-from-scratch validation exists. Do not extrapolate to "BT V5 is a working replacement architecture" — it is a working single-layer retrofit.
- **Multi-layer swap works up to 3 layers** ({L7, L10, L12} at +1.38%), but degrades past that for pure Focus.

---

## Known issues

- Real-mode benchmarks require GPU with ~8 GB VRAM for Qwen3-0.6B in fp32; bf16 fits in 4.3 GB (GTX 1650 dev target).
- Mock mode uses synthetic embeddings for CI without GPU.
- Legacy `DualHeadPlateauAttention` (SIRI) returns `(output, A_low, A_high)`; SDOT dual-head tests were moved to legacy.
- `power_diagrams.py` ψ bias absorption behavior is expected math, documented in `test_power_diagram_psi_is_applied`.

---

## Migration history

- **June 2026** — SDOT v3/v4 removed. Moved to `docs/legacy/sdot_v3_v4/`:
  - `models/sdot_attention.py`, `models/sdot_attention_v4.py`, `models/qwen3_gqa_bubble_wrapper.py`
  - `scripts/inject_sdot_qwen.py`
  - 8 SDOT-dependent test files (171 tests archived)
- **June 2026** — Added: `deltanet_attention.py`, `siri_postprocess.py`, `siri_soft.py`, `power_diagrams.py`, `hybrid_attention.py`.
- **June 2026** — SIRI-Soft variants introduced (`soft`, `chiller`, `sparse`) to preserve peakedness. Best variant: soft blend, PPL 26.76 (still failed the 2% gate).
- **July 2026** — **Focus Bubble V5** introduced. First configuration to pass the gate: FocusDeltaNet L7 λ=0.3 at +0.16% PPL.
- **July 2026** — arXiv paper drafted (`paper/main.tex`), blog post published (`docs/blog_post.md`), GitHub Pages live with PDF.

Test count trajectory: 564 (pre-SDOT-removal) → 393 (post-cleanup) → 462 (post-SIRI-Soft) → **490 (current, post-Focus-Bubble)**. Note: earlier docs (blog post, arXiv paper draft) cite "501 passing" — that was on the dev box with GPU + cached Qwen3 and pre-NumPy-2.4. The bare-env, reproducible count is 468/490.

Full migration rationale: [`docs/decisions/2026-06-27-sota-replacement-siri-preserved.md`](docs/decisions/2026-06-27-sota-replacement-siri-preserved.md).

---

## Working with this repo

1. **Read `IMPORTANTE/BT-V5_06_focus_bubble.md` first** if you touch anything attention-related — that document is the current architectural source of truth.
2. **Read `IMPORTANTE/BT-V5_05_protocol_positioning.md`** before running benchmarks — it defines the gate, thresholds, and honest positioning.
3. **The gate is the arbiter.** ΔPPL > 2% on Qwen3-0.6B is a fail, no matter how good $R_{\text{eff}}$ / concentration ratio / anisotropy look.
4. **When you break an invariant, say so.** The Focus Bubble breaks I1 and reshapes I2 — both are documented above and in the paper. Silent invariant breakage is worse than a documented one.
5. **Commit messages follow Conventional Commits** — see recent history for style.
6. **AGENTS.md and README.md are kept in sync manually.** If you change the architecture, update both.
