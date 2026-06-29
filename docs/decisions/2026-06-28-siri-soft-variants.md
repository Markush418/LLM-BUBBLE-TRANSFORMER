# SIRI-Soft: Peakedness-Preserving Doubly-Stochastic Variants

**Date**: June 28, 2026
**Author**: LLM-BUBBLE Project
**Status**: Implemented and validated

## Problem Statement

Classical SIRI (Sinkhorn-Knopp doubly-stochastic) destroys attention peakedness. The
doubly-stochastic constraint forces every entry to be approximately 1/N on average,
which eliminates the natural sparse structure that softmax attention provides.

**Empirical evidence** (Qwen3-0.6B, WikiText-2 perplexity):

| Mode | PPL | Δ PPL |
|------|-----|-------|
| Baseline (softmax) | 23.37 | 0.00 |
| Pure SIRI (λ=0) | 568.21 | +545 |
| Classical SIRI (λ=0.5) | 30.14 | +6.77 |

The classical SIRI mode is catastrophic — the doubly-stochastic constraint is too
restrictive for language modeling.

## Mathematical Analysis

Given attention matrix A from softmax: `A[i,j] = exp(q_i·k_j/√d) / Σ_k exp(...)`.

Classical SIRI produces doubly-stochastic A' via Sinkhorn:
```
A' = Sinkhorn(scores)  # both row AND column sums = 1
```

The doubly-stochastic constraint implies `Σ_{i,j} A'[i,j] = N` (not N² as in softmax).
Therefore the average entry is 1/N, regardless of how peaked the input scores were.

This destroys the property `max(A[i,:]) ≈ 1` (one-hot attention) that softmax provides
naturally.

## Solution: Three SIRI Variants

### Variant 1: Soft Blend (SpikeFormer-style)

**Formula**: `A_soft = (1-α) · softmax + α · Sinkhorn(scores)`

**Rationale**: 
- Softmax contributes peakedness via the unconstrained numerator.
- SIRI contributes column-uniformity via the Sinkhorn step.
- Blend ratio α controls the strength of doubly-stochasticity.

**Default**: α = 0.3 (mostly softmax, gentle SIRI regularization).

**Result**: PPL 26.76 at λ=0.5, layer 3, Qwen3-0.6B.

### Variant 2: Chiller (Sinkhorn with sharpening)

**Formula**: `A_chill = Sinkhorn(scores · β)`

**Rationale**:
- Multiply scores by β > 1 BEFORE Sinkhorn → sharpened kernel.
- After Sinkhorn, row sums = 1 (not col sums).
- Preserves peakedness much better than classical.

**Default**: β = 5.0.

**Result**: PPL 39.39 at λ=0.5, layer 3. Less effective than soft blend.

### Variant 3: Sparse (ReLU + Sinkhorn)

**Formula**: `A_sparse = Sinkhorn(ReLU(-C/ε))`

**Rationale**:
- Use geometric cost C = ||Q-K||², but ReLU on `−C/ε` zeros out far-away tokens.
- Sinkhorn then enforces doubly-stochasticity only on remaining entries.
- Produces very sparse doubly-stochastic matrices.

**Default**: ε = 0.1.

**Result**: Sparse peakedness (peak ratio >100× softmax), but tighter constraint.

## Implementation

File: `experiments/siri_soft.py` (NumPy, ~290 lines)

```python
from siri_soft import siri_soft_blend, siri_chiller, siri_sparse

# Variant 1: Soft blend
A_soft = siri_soft_blend(scores, alpha=0.3, tau_iters=5)

# Variant 2: Chiller
A_chill = siri_chiller(scores, beta=5.0, tau_iters=5)

# Variant 3: Sparse
A_sp = siri_sparse(scores, tau_iters=20)
```

## Integration with Qwen3 Wrapper

File: `experiments/qwen3_hybrid_gqa_wrapper.py`

The wrapper accepts `siri_mode` parameter:

```python
wrapper = Qwen3HybridGQABubbleWrapper(
    original_attn=original_attn,
    epsilon=0.01,
    lam=0.5,
    siri_mode="soft",   # classical | chiller | sparse | soft
    siri_alpha=0.3,     # only used in soft mode
    siri_beta=5.0,      # only used in chiller mode
)
```

## Validation

### Unit Tests (19 tests, all passing)

`tests/test_siri_soft.py`:
- Shape and dtype correctness
- Row-stochasticity (row sums = 1.0)
- Peakedness (Chiller/Sparse > softmax)
- Causal mask preservation
- Numerical stability (no NaN/Inf at high β)
- Torch tensor compatibility

### Perplexity Benchmark

`results_real/perplexity_L3_siri/`:
- Soft blend: PPL 26.76 (best)
- Classical SIRI: PPL 30.14
- Chiller: PPL 39.39

## Recommendation

**Use `siri_mode="soft"` with α ∈ [0.3, 0.7] as the default for new experiments.**
This preserves the peakedness of softmax while softly enforcing doubly-stochastic
regularization. The pure SIRI mode (siri_mode="classical") should be reserved for
cases where strict doubly-stochasticity is required (e.g., optimal transport
interpretations).

## Related Work

- **Sinkformers** (Sander et al. 2021, arXiv:2110.11773) — original SIRI formulation
- **SpikeFormer** (Li et al. 2024) — spike-and-slab attention with peakedness
- **Gumbel-Sinkhorn** (Mena et al. 2018) — differentiable approximate doubly-stochastic

## Future Work

1. **Gumbel-Sinkhorn variant** — add Gumbel noise for stochastic exploration
2. **Sparse-Max attention** — Martins & Astudillo 2016, true sparse softmax
3. **Adaptive α** — learn α end-to-end via gradient
4. **Multi-step Sinkhorn** with sparse masking — sparse pattern emerges from iterations

---

*Decision record — June 28, 2026*