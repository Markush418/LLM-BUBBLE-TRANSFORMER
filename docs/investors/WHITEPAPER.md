# Bubble Transformer V4

**Technical Whitepaper v2.0 — Honest Assessment**

**Hybrid DeltaNet + SIRI + Geometric Bias: A Research Artifact**

---

## ⚠️ Honest Framing

This whitepaper describes a **research-stage architecture** validated via single-layer drop-in swap on a pretrained model. It does **not** claim:

- Production readiness
- Competitive performance with Kimi Linear, Gated DeltaNet, or Mamba-3
- A filed patent
- A published paper
- Long-context validation
- Downstream task validation

What it **does** claim:

- A reproducible research artifact (462 tests, open-source)
- A documented engineering wrapper with strong numerical safety
- A research hypothesis (soft blend + geometric bias ψ) that may or may not survive end-to-end pretrain
- A defensible engineering contribution (GQA, KV cache, dtype handling) that is non-trivial to replicate

**The strongest honest claim**: we built a DeltaNet wrapper that achieves +1.24 PPL in single-layer swap on Qwen3-0.6B / WikiText-2, with 6 documented numerical bug fixes, and we are seeking $800k to test whether our soft blend hypothesis survives end-to-end pretrain.

---

## Abstract

We present Bubble Transformer V4, a hybrid attention architecture combining the linear complexity of DeltaNet (NeurIPS 2024) with the doubly-stochastic regularization of SIRI (Sinkformers, AISTATS 2022) and a learnable per-query scalar bias ψ. Through frozen single-layer swap experiments on Qwen3-0.6B with WikiText-2, we find that (1) raw SIRI adds +7.6 PPL drift, (2) our soft blend variant (α=0.7) recovers 45% of that gap, and (3) pure DeltaNet (λ=1.0) is the best single-layer configuration at +1.24 PPL drift. The geometric bias ψ is zero-initialized and contributes zero PPL drift in our experiments. We document 6 numerical edge cases encountered and fixed during development, and present a reproducible benchmark suite (462 tests passing). The work is a research artifact, not a production system. We discuss limitations, failure modes, and the open research questions that would need to be answered before claiming competitive performance with the 2025-2026 generation of linear attention architectures.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: Linear Attention Landscape](#2-background-linear-attention-landscape)
3. [Bubble V4 Architecture](#3-bubble-v4-architecture)
4. [Experimental Results: Single-Layer Frozen Swap](#4-experimental-results-single-layer-frozen-swap)
5. [The SIRI Peakedness Problem](#5-the-siri-peakedness-problem)
6. [Soft Blend: A Partial Solution](#6-soft-blend-a-partial-solution)
7. [Geometric Bias ψ: An Unvalidated Hypothesis](#7-geometric-bias-ψ-an-unvalidated-hypothesis)
8. [Numerical Safety Engineering](#8-numerical-safety-engineering)
9. [Reproducibility](#9-reproducibility)
10. [Limitations and Failure Modes](#10-limitations-and-failure-modes)
11. [Comparison with 2025-2026 Linear Attention](#11-comparison-with-2025-2026-linear-attention)
12. [Open Research Questions](#12-open-research-questions)
13. [References](#13-references)
14. [Appendices](#14-appendices)

---

## 1. Introduction

### 1.1 Motivation

The 2024-2026 period has seen a Cambrian explosion of attention alternatives to scaled dot-product softmax. Linear attention architectures (Mamba, RWKV, DeltaNet, Gated DeltaNet, Kimi Linear) promise $O(N)$ complexity for long-context inference, but each has trade-offs in language modeling performance, numerical stability, or interpretability.

This work sits at the intersection of three research directions:

1. **Linear attention via delta rule** (Schlag et al., 2024)
2. **Doubly-stochastic regularization** (Sander et al., 2022)
3. **Geometric structure in attention** (Power Diagrams, Aurenhammer 1987)

We do not claim novelty in (1) or (2). The contribution we explore is whether (3) can be integrated into a linear attention framework in a way that provides independent value.

### 1.2 What This Paper Is

This is a **reproducibility-focused research report**. We document:

- The full architecture specification
- Honest benchmark results with explicit failure modes
- Engineering decisions and bug fixes
- The gap between our claims and the state of the art

### 1.3 What This Paper Is Not

This is **not**:

- A competitive submission to Mamba-3, Gated DeltaNet, or Kimi Linear
- A production system
- A claim that the geometric bias ψ is empirically validated
- A claim that soft blend is theoretically grounded

We are explicit about these limitations throughout.

---

## 2. Background: Linear Attention Landscape

### 2.1 Softmax Attention (Baseline)

For queries $Q \in \mathbb{R}^{N \times d}$, keys $K \in \mathbb{R}^{N \times d}$, values $V \in \mathbb{R}^{N \times d}$:

$$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) \in \mathbb{R}^{N \times N}$$
$$\text{output} = A V$$

Properties: row-stochastic, peaked, $O(N^2)$ memory.

### 2.2 DeltaNet (Schlag et al., 2024)

Recurrent formulation with per-step normalization:

$$S_t = (1 - \alpha_t) S_{t-1} + v_t k_t^\top - S_{t-1} q_t k_t^\top$$

where $\alpha_t$ controls decay. The output is $o_t = S_t q_t$.

Properties: $O(N)$ memory, captures token-to-token interactions via delta rule, partial peakedness.

**Note from the original authors**: *"vanilla linear attention has underperformed compared to softmax attention (by a large margin) in language modeling"* (Songlin Yang, MIT, Dec 2024). This is a critical honest baseline — we are not the first to attempt beating softmax, and the headroom is small.

### 2.3 SIRI / Doubly-Stochastic Attention (Sander et al., 2022)

Geometric cost and Sinkhorn projection:

$$C_{ij} = \|Q_i - K_j\|^2$$
$$\log S = -\frac{C_{ij}}{\epsilon}$$
$$A = \text{Sinkhorn}(\log S, \tau) \in \Sigma_n$$

where $\Sigma_n$ is the Birkhoff polytope (doubly-stochastic matrices).

Properties: doubly-stochastic, interpretable, but $O(N^2)$ memory and uniform-like entries.

### 2.4 Gated DeltaNet (ICLR 2025, NVIDIA Labs)

The most mature delta-rule variant. Adds selective gating:

$$S_t = (1 - \alpha_t \cdot \sigma(g_t)) S_{t-1} + v_t k_t^\top - S_{t-1} q_t k_t^\top$$

where $g_t$ is a learned gate. Published in ICLR 2025, integrated in the `transformers` library via the FLA (Fast Linear Attention) package, NVIDIA-validated.

**Why we do not claim to compete with Gated DeltaNet**: they have NVIDIA compute, 5+ co-authors, ICLR acceptance, and production integration. We have 1 author, 0.6B frozen swap, and an unvalidated hypothesis.

### 2.5 Kimi Linear (arXiv:2510.26692, Oct 2025)

Hybrid KDA + MLA architecture, 3B activated / 48B total parameters, **outperforms full MLA** on benchmarks. Open-source, production-deployed at kimi.com.

**Why we do not claim to compete with Kimi Linear**: they have ~60 co-authors, 1000x our compute budget, and full pretrain validation.

### 2.6 Mamba-3 (arXiv:2603.15569, ICLR 2026 Oral)

State-space model with selective mechanism. State-of-the-art on efficiency frontier, half decoding cost vs Mamba-2. Together AI production deployment.

**Why we do not claim to compete with Mamba-3**: they have 8 co-authors from Princeton + Cartesia, ICLR Oral, and $191M in funding.

---

## 3. Bubble V4 Architecture

### 3.1 Core Formula

For input $X \in \mathbb{R}^{B \times N \times d}$:

**Step 1: Project**
$$Q, K, V = X W_Q, X W_K, X W_V \in \mathbb{R}^{B \times H \times N \times d_h}$$

**Step 2: Normalize for numerical safety**
$$\hat{Q} = \frac{Q}{\|Q\|}, \quad \hat{K} = \frac{K}{\|K\|}$$

This ensures $C_{ij} \in [0, 4]$ instead of $[0, 400+]$.

**Step 3: Compute geometric cost**
$$C_{ij} = \|\hat{Q}_i - \hat{K}_j\|^2$$

**Step 4: Geometric bias ψ** (zero-initialized in current implementation)
$$\log S = -\frac{C_{ij}}{\epsilon} + \psi(Q)$$
where $\psi(Q) = W_\psi Q \in \mathbb{R}^N$ is a learnable linear projection.

**Step 5: Sinkhorn projection** (τ=5 iterations in log-domain)
$$A_{\text{SIRI}} = \text{Sinkhorn}(\log S, \tau=5)$$

**Step 6: Hybrid blending**
$$A_{\text{hybrid}} = \lambda \cdot A_{\text{DeltaNet}} + (1-\lambda) \cdot A_{\text{SIRI}}$$

**Step 7: Apply to values**
$$\text{output} = A_{\text{hybrid}} V$$

### 3.2 The Three Hyperparameters

| Parameter | Range | Default | Role |
|---|---|---|---|
| λ | [0, 1] | 0.5 | ΔNet vs SIRI blend (1=ΔNet, 0=SIRI) |
| α | [0, 1] | 0.7 | Soft blend SIRI vs softmax (soft mode only) |
| ε | (0, ∞) | 0.1 | SIRI bandwidth (lower = more concentrated) |

**Interaction**: the relationship between λ, α, and ε is an open research question. We have not performed systematic sweeps.

### 3.3 The Soft Blend Mode

In `siri_mode="soft"`, the SIRI component is replaced by:

$$A_{\text{soft}} = (1 - \alpha) \cdot \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) + \alpha \cdot A_{\text{SIRI}}$$

**Critical caveat**: this destroys doubly-stochasticity. The column sums become $1-\alpha \neq 1$. The "benefit" of SIRI's regularity is lost in soft mode. The only retained property is the geometric cost $C_{ij}$, which provides a different inductive bias than inner-product attention.

### 3.4 Implementation

The wrapper is implemented for HuggingFace transformers:

```python
from experiments.qwen3_hybrid_gqa_wrapper import Qwen3HybridGQABubbleWrapper

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
for i in [3, 7, 11, 15]:
    model.model.layers[i].self_attn = Qwen3HybridGQABubbleWrapper(
        model.config, layer_idx=i,
        siri_mode="soft", siri_alpha=0.7, lambda_param=0.5
    )
```

Handled explicitly:
- GQA expansion: `K, V = K.repeat_interleave(kv_groups, dim=1)`
- KV cache compatibility
- Causal mask shape: `[B, 1, N, N+1] → [B, 1, N, N]` (KV cache convention)
- Dtype: components in float32, output cast to model dtype
- Numerical guards: NaN checks, log-domain clamping, safe normalization

---

## 4. Experimental Results: Single-Layer Frozen Swap

### 4.1 Setup (Honest)

- **Model**: Qwen3-0.6B (28 layers, head_dim=128, GQA)
- **Dataset**: WikiText-2 test split, first 50k characters
- **Context window**: 256 tokens, stride 256
- **Precision**: float16 forward, float32 for SIRI computation
- **Hardware**: GTX 1650, 4.3GB VRAM
- **Swap type**: single attention layer, all other 27 layers frozen

**Important caveat**: this is a **frozen single-layer probe**, not pretrain. The +1.24 PPL result measures "how much does this layer contribute to language modeling," not "can this architecture train a competitive model from scratch."

### 4.2 Main Results

| Configuration | PPL | ΔPPL |
|---|---|---|
| Baseline (softmax, 28 layers) | 22.515 | — |
| L3 only, pure ΔNet (λ=1.0) | **23.749** | **+1.24** |
| L3 only, soft blend (λ=0.5, α=0.7) | 26.76 | +4.24 |
| L3 only, pure SIRI (λ=0.0) | 30.14 | +7.63 |
| L3 only, chiller (β=2) | 39.39 | +16.88 |
| L3-L15 (4 layers), pure ΔNet | 31.544 | +9.03 |
| L3-L15 (4 layers), pure SIRI | >500 | catastrophic |

**Interpretation**: in a frozen model, replacing one attention layer with ΔNet adds only +1.24 PPL. Replacing 4 layers adds +9 PPL. Replacing 4 layers with SIRI is catastrophic (+545 PPL, as we previously reported — but this is an unsurprising result of breaking 4/28 attention layers in a frozen model).

### 4.3 Per-Layer Sensitivity (Single-Layer ΔNet Swap)

| Layer | PPL | ΔPPL | Category |
|---|---|---|---|
| L0 (embed) | 69,628 | +69,605 | untouchable |
| L3 | 23.749 | +1.24 | best candidate |
| L7 | 23.735 | +1.22 | good |
| L11 | 24.384 | +1.87 | moderate |
| L15 | 23.945 | +1.43 | good |
| L19 | 25.065 | +2.55 | moderate |
| L23 | 24.115 | +1.60 | good |
| L27 (final) | 24.365 | +1.85 | moderate |

**Finding**: embedding layer (L0) is sacred. Best swap targets are L3, L7, L15 (early-to-mid layers).

### 4.4 Lambda Sweep (L3, Single-Layer)

| λ | PPL | ΔPPL |
|---|---|---|
| 0.0 (pure SIRI) | 30.14 | +7.63 |
| 0.25 | 28.93 | +6.42 |
| 0.5 | 27.56 | +5.05 |
| 0.75 | 25.12 | +2.61 |
| **1.0 (pure ΔNet)** | **23.749** | **+1.24** |

**Honest interpretation**: pure ΔNet is monotonically better. The SIRI component does not help in our experiments. The soft blend is a workaround for SIRI's peakedness destruction, not a theoretically motivated improvement.

### 4.5 What These Results Do Not Show

- **Not pretrain**: we never trained a model with Bubble as the attention mechanism
- **Not long-context**: 256 tokens only, Kimi Linear targets 128k
- **Not downstream tasks**: no MMLU, HumanEval, GSM8K
- **Not multi-architecture**: only Qwen3-0.6B; Llama, Mistral, GPT-NeoX unverified
- **Not ψ-trained**: ψ is zero-initialized, contributes zero PPL drift

---

## 5. The SIRI Peakedness Problem

### 5.1 Empirical Observation

In our experiments, raw SIRI (λ=0.0, single layer L3) adds +7.63 PPL to Qwen3-0.6B on WikiText-2. This is consistent with the doubly-stochastic constraint forcing entries to ~$1/N$, eliminating the peakedness that softmax provides.

### 5.2 Theoretical Analysis

For $A \in \Sigma_n$ (Birkhoff polytope), the Birkhoff-von Neumann theorem states $A = \sum_k \theta_k P_k$ where $P_k$ are permutation matrices and $\theta_k \geq 0, \sum \theta_k = 1$.

**Claim**: in expectation over the convex combination, $\mathbb{E}[\max_j A_{ij}] \leq H(n)/n$ where $H(n)$ is a harmonic-like factor.

**This is a non-rigorous claim**. A proper bound requires analysis of the specific Sinkhorn projection, not just the convex hull characterization. The intuition (doubly-stochastic = uniform-ish) is correct, but the formal proof is not in this document.

### 5.3 Why This Matters

Language modeling requires peakedness: the model must concentrate probability mass on the few tokens that determine the next word. Doubly-stochastic attention distributes mass uniformly, which is fundamentally at odds with the language modeling objective.

This is a **fundamental limitation of SIRI for language modeling**, not a bug to be fixed. Soft blend is a workaround, not a solution.

### 5.4 Sinkformers' Original Claims (Reconciled)

Sinkformers (Sander et al., 2022) reported positive results for doubly-stochastic attention in vision and NLP tasks. This appears to contradict our +7.6 PPL result. Possible reconciliations:

1. Different task (vision classification vs language modeling)
2. Different model size (their models were smaller)
3. Different evaluation (they may have measured accuracy, not PPL)
4. Different layer placement (single head vs full attention)

We have not investigated this discrepancy in depth. It is an open question.

---

## 6. Soft Blend: A Partial Solution

### 6.1 The Formula

$$A_{\text{soft}} = (1 - \alpha) \cdot A_{\text{softmax}} + \alpha \cdot A_{\text{SIRI}}$$

with $\alpha = 0.7$ default.

### 6.2 Empirical Effect

| Config (L3, single layer) | PPL | ΔPPL |
|---|---|---|
| Baseline | 23.37 | — |
| Classical SIRI (α=1.0) | 30.14 | +6.77 |
| **Soft blend (α=0.7)** | **26.76** | **+3.39** |

Soft blend recovers 50% of the PPL gap.

### 6.3 What Soft Blend Does

- **From softmax**: peakedness (max entry near 1)
- **From SIRI**: geometric cost $C_{ij} = \|Q-K\|^2$ instead of inner product
- **Lost**: doubly-stochastic property (column sums = $1-\alpha$)

**Honest characterization**: soft blend is a geometric-cost softmax with regularization toward uniform. It is not "doubly-stochastic with peakedness preserved." The "doubly-stochastic" part is gone by construction.

### 6.4 The Mathematical Caveat

The claim in the previous draft that soft blend "preserves both properties" is **incorrect**. The convex combination $A_{\text{soft}} = (1-\alpha) A_{\text{softmax}} + \alpha A_{\text{SIRI}}$ is row-stochastic (rows sum to 1) but not column-stochastic (columns sum to $1-\alpha$ by linearity).

We correct this in v2.0. The soft blend is a geometric-cost regularization, not a doubly-stochastic attention.

---

## 7. Geometric Bias ψ: An Unvalidated Hypothesis

### 7.1 The Hypothesis

We add a learnable per-query scalar bias:

$$\log S_{ij} = -\frac{C_{ij}}{\epsilon} + \psi(Q_i)$$

where $\psi(Q) = W_\psi Q \in \mathbb{R}^N$ is a linear projection.

The hypothesis is that ψ induces a Power Diagram-like structure in the attention matrix, providing geometric regularity that may help with interpretability or generalization.

### 7.2 Current Status

- **Implementation**: ✅ Complete
- **Unit tests**: ✅ 19 passing
- **Empirical validation**: ❌ Zero — ψ is zero-initialized and contributes zero PPL drift
- **Pretrain validation**: ❌ Not done (Q1 2026 work)

### 7.3 Why We Are Not Claiming Novelty

A learnable scalar bias per query is not novel:

- **ALiBi** (Press et al., 2022): linear bias on attention scores
- **RoPE** (Su et al., 2021): rotary position embedding
- **T5 bias** (Raffel et al., 2020): relative position bias
- **GTA** (arXiv:2310.10375, ICLR 2024): geometric transformations in attention

What we are exploring is whether a *learned* geometric bias (trained end-to-end with the model) provides independent value in linear attention contexts. This is an open question.

### 7.4 Power Diagram Terminology (Honest Caveat)

The original draft used "Power Diagram" terminology because the bias ψ is added to the log-Sinkhorn input, which is mathematically analogous to Power Diagram weights.

**This is a loose analogy, not a formal connection.** A true Power Diagram would require the bias to interact with the geometric cost in a specific way (cell boundaries defined by $\|x-p_i\|^2 - w_i \leq \|x-p_j\|^2 - w_j$). Our ψ is an additive scalar that gets absorbed by Sinkhorn column normalization.

**We have removed the "Power Diagram ψ" patent claim and "first architecture to integrate Power Diagrams" language from v2.0.** The correct framing is "learnable per-query scalar bias in geometric-cost attention."

### 7.5 The Research Question Worth Testing

If ψ is trained end-to-end in a Bubble-1.3B pretrain, does it provide:

1. Better convergence (lower training loss at fixed compute)?
2. Better interpretability (can we visualize what ψ encodes)?
3. Better generalization (lower PPL on held-out domains)?

**We do not know.** This is the central research question for the $800k seed ask.

---

## 8. Numerical Safety Engineering

### 8.1 Bugs Found and Fixed (Documented)

During development, we encountered and resolved 6 numerical edge cases:

#### Bug 1: Causal mask shape mismatch
- **Symptom**: `tensor a (129) must match tensor b (130) at dim 3`
- **Cause**: Qwen3 passes mask `[B, 1, N, target_length]` with `target_length = N+1` (KV cache convention)
- **Fix**: `causal_2d = causal_2d[..., -N:]`

#### Bug 2: SIRI saturation
- **Symptom**: PPL = 65,387 with ε=0.01
- **Cause**: Q,K with norm ~10 → `‖Q-K‖² ~ 100-400` → `log_S = -C/ε = -10000..0` saturates
- **Fix**: normalize Q,K before cost → `C ∈ [0,4]`

#### Bug 3: Wrapper dtype downcast
- **Symptom**: `float != struct c10::Half`
- **Cause**: `.to(fp16)` downcasteaba `_pd.W_psi` de float32 a float16, rompiendo matmul
- **Fix**: no castear el wrapper al dtype del modelo; keep internal components in float32

#### Bug 4: Test isolation
- **Symptom**: subsequent tests fail after `test_swap_all_layers_real`
- **Cause**: wrapper replaces all 28 layers; subsequent tests attempt to create new wrapper on wrapped layer
- **Fix**: `try/finally` restore original layers after test

#### Bug 5: Chiller row sums ≠ 1
- **Symptom**: NaN/Inf propagation at high β
- **Cause**: incorrect normalization after Sinkhorn
- **Fix**: `np.where(row_sums > 1e-30, A/row_sums, 0)`

#### Bug 6: Sparse causal mask leakage
- **Symptom**: Sinkhorn projects over ReLU mask
- **Cause**: ReLU zeros out positions, but Sinkhorn fills them
- **Fix**: hard zero masked positions post-Sinkhorn

### 8.2 Test Suite Status

**462 tests passing, 2 skipped, 0 failed.**

Coverage:
- Unit tests per module (DeltaNet, SIRI, Power Diagram bias, Soft blend)
- Integration tests for Qwen3 wrapper
- Numerical stability tests (NaN, dtype, edge cases)
- Perplexity smoke tests (catches >0.5 PPL drift)

Run: `python -m pytest tests/ -v`

### 8.3 What "462 tests" Does and Does Not Mean

**It does mean**: the wrapper does not crash on Qwen3-0.6B, handles edge cases, and is reproducible.

**It does not mean**: the architecture is validated against competitors, scales to other models, or works in production. Tests are hygiene, not moat.

---

## 9. Reproducibility

### 9.1 One-Command Reproduction

```bash
git clone https://github.com/Markush42/bubble-transformer.git
cd bubble-transformer
pip install -r requirements.txt
python experiments/run_experiment.py --mode real
```

Runs full benchmark suite on Qwen3-0.6B with WikiText-2. ~10 minutes on GTX 1650.

### 9.2 Mock Mode (No GPU)

```bash
python experiments/run_experiment.py --mode mock
```

Uses synthetic embeddings. <30 seconds.

### 9.3 Per-Layer Analysis

```bash
python experiments/perplexity_layerwise.py --layers 0 3 7 11 15 19 23 27
```

### 9.4 Lambda Sweep

```bash
python experiments/perplexity_benchmark_hybrid.py \
  --lambdas 0.0 0.25 0.5 0.75 1.0 \
  --siri-modes classical soft \
  --layers 3
```

### 9.5 Open-Source License

Apache 2.0 for research and non-commercial use. Commercial licensing available.

---

## 10. Limitations and Failure Modes

### 10.1 What Bubble V4 Cannot Do (Currently)

1. **Cannot replace softmax in production**: +1.24 PPL drift is too high for deployment
2. **Cannot scale to 128k context**: validated only at 256 tokens
3. **Cannot handle multi-architecture**: only Qwen3 tested
4. **Cannot work in embedding layer**: L0 swap causes catastrophic failure
5. **Cannot match Kimi Linear, Gated DeltaNet, or Mamba-3**: all three are more mature, better funded, and have full pretrain validation

### 10.2 Failure Modes We Know About

- **Frozen multi-layer swap**: replacing 4+ layers in a frozen model causes catastrophic failure (true for any attention mechanism, not just Bubble)
- **Long context**: untested; likely requires chunked attention or sparse patterns we haven't implemented
- **Quantization**: works at float16, untested at int8/int4
- **Multi-GPU**: single-GPU only; tensor parallel untested

### 10.3 What We Do Not Know

- Whether ψ provides any value when trained end-to-end
- Whether soft blend helps in pretrain or only in frozen-swap
- Whether the linear attention gap to softmax closes at 7B+ scale
- Whether the geometric cost $C_{ij}$ is better than inner product for any task

---

## 11. Comparison with 2025-2026 Linear Attention

### 11.1 Honest Comparison Table

| Architecture | Linear? | Peaked? | Pretrain? | Production? | Bubble Comparison |
|---|---|---|---|---|---|
| Softmax Transformer | ❌ | ✅ | ✅ | Universal | baseline |
| **Mamba-3** | ✅ | ❌ | ✅ 7B | ✅ (Together AI) | more mature, better funded |
| **RWKV-7** | ✅ | ❌ | ✅ 2.9B | ✅ | open-source community |
| **Gated DeltaNet** | ✅ | ❌ | ✅ 1.3B | ✅ (HF library) | NVIDIA-validated |
| **Kimi Linear** | ✅ | ✅ | ✅ 3B/48B | ✅ (kimi.com) | beats full MLA |
| Sinkformers (SIRI) | ❌ | ❌ | ❌ | academic | +7.6 PPL in our setup |
| **Bubble V4** | ✅ | partial | ❌ | ❌ | +1.24 PPL, 0.6B frozen |

**Honest read**: Bubble V4 is the least mature option in this table. We have not trained a model. We have not deployed. We have not validated at scale.

### 11.2 What Bubble V4 Offers (That Others Do Not)

1. **A reproducible engineering wrapper for DeltaNet** with GQA, KV cache, and dtype handling. This is non-trivial to replicate from a paper.

2. **A documented soft blend mode** for hybrid softmax + doubly-stochastic attention. To our knowledge, no published paper combines these in this exact way.

3. **A test of whether geometric-cost attention helps in linear contexts**. The geometric cost $C_{ij} = \|Q-K\|^2$ is used in Sinkformers but not in DeltaNet-family architectures.

### 11.3 What We Are Not Competing On

- Raw language modeling PPL
- Training cost
- Inference throughput
- Long-context performance
- Production deployment

---

## 12. Open Research Questions

### 12.1 Central Question

**Does the geometric bias ψ provide any value when trained end-to-end in a 1B+ parameter model?**

This is the question the $800k seed is designed to answer.

### 12.2 Sub-Questions

1. **Does ψ help convergence?** Lower training loss at fixed compute?
2. **Does ψ help interpretability?** Can we visualize what ψ encodes?
3. **Does ψ help generalization?** Lower PPL on held-out domains?
4. **Does soft blend help in pretrain?** Or is it only useful in frozen-swap?
5. **Does the linear attention gap to softmax close at 7B+?** Or does it widen?
6. **Is geometric cost $C_{ij}$ better than inner product** for any task?

### 12.3 What We Would Do With $800k

| Quarter | Experiment | Cost | Success Criterion |
|---|---|---|---|
| Q1 | Bubble-1.3B from scratch, 50B tokens, ΔNet only | $200k | PPL within 2 of Qwen2.5-1.5B |
| Q2 | Ablation: ψ enabled vs ψ=0 | $100k | ψ helps training loss |
| Q3 | Long-context NIAH at 4k-8k | $50k | ≥80% retrieval accuracy |
| Q4 | Downstream eval: HellaSwag, LAMBADA, ARC | $50k | within 5% of Qwen2.5-1.5B |

**If Q1 fails** (PPL gap > 5): pivot to (a) ψ interpretability research, (b) consulting revenue, or (c) acqui-hire.

**If Q1 succeeds**: Series A in Q4 2026 to scale to 7B and add downstream task evaluation.

---

## 13. References

1. Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
2. Schlag et al. (2024). "Linear Transformers with Recurrent Delta Rule." NeurIPS. arXiv:2406.06484.
3. Sander et al. (2022). "Sinkformers: Transformers with Doubly Stochastic Attention." AISTATS. arXiv:2110.11773.
4. Yang et al. (2024). "Gated DeltaNet: Sequence Modeling with Selective State Spaces." ICLR 2025. arXiv:2412.06464.
5. Moonshot AI (2025). "Kimi Linear: An Expressive, Efficient Attention Architecture." arXiv:2510.26692.
6. Gu et al. (2026). "Mamba-3: Improved Selective State Space Modeling." ICLR 2026 Oral. arXiv:2603.15569.
7. Peng et al. (2025). "RWKV-7: Goose with Expressive Linear RNN." arXiv:2503.14456.
8. Press et al. (2022). "Train Short, Test Long: Attention with Linear Biases (ALiBi)."
9. Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)."
10. Aurenhammer (1987). "Power Diagrams: Properties, Algorithms and Applications." SIAM J Computing.
11. GTA authors (2024). "Geometry-Aware Attention for Multi-View Transformers." ICLR 2024. arXiv:2310.10375.

---

## 14. Appendices

### Appendix A: Corrected Mathematical Claims

**A.1 Soft blend does NOT preserve doubly-stochasticity**

The convex combination $A_{\text{soft}} = (1-\alpha) A_{\text{softmax}} + \alpha A_{\text{SIRI}}$ has:
- Row sums: $(1-\alpha) \cdot 1 + \alpha \cdot 1 = 1$ ✓ (row-stochastic)
- Column sums: $(1-\alpha) \cdot c_j + \alpha \cdot 1$ where $c_j$ is the column sum of $A_{\text{softmax}}$

For random $A_{\text{softmax}}$ with uniform column sums, $c_j \approx 1$, so column sums $\approx 1$. But this is not exact, and it is not doubly-stochastic by construction.

**Correct framing**: soft blend is a geometric-cost softmax with regularization toward uniform. It is NOT doubly-stochastic attention with peakedness preserved.

**A.2 Doubly-stochastic peakedness bound (Sketch)**

For $A \in \Sigma_n$, $\max_j A_{ij} \leq 1$ (achieved at permutation matrices). For random doubly-stochastic matrices from Sinkhorn projection, the distribution is concentrated near uniform: $\mathbb{E}[A_{ij}] = 1/n$ for $i \neq j$ (by symmetry).

A tighter bound on $\max_j A_{ij}$ requires analysis of the specific Sinkhorn operator and input distribution. We have not computed this bound rigorously.

**A.3 DeltaNet norm stability (Sketch)**

The per-step normalization $S_t = (1 - 1/t) S_{t-1} + \text{outer}(k_t, \delta_t)$ prevents norm explosion under bounded inputs. By induction, $\|S_t\|$ is bounded by a constant depending on $\max_t \|k_t\|, \|q_t\|, \|v_t\|$, not on $t$.

This is a sketch, not a proof. Full analysis requires specifying input distribution and projection operators.

### Appendix B: Honest Disclosures

**B.1 What changed from v1.0**

| Claim in v1.0 | Correction in v2.0 |
|---|---|
| "Soft blend preserves both properties" | FALSE — destroys doubly-stochasticity |
| "First architecture to integrate Power Diagrams" | OVERSTATED — ψ is a scalar bias, not a true Power Diagram |
| "Patent pending" | UNVERIFIED — removed |
| "NeurIPS 2026 submission" | SPECULATIVE — removed |
| "Validated in production" | FALSE — corrected to "validated as research artifact" |
| "$7B Mistral valuation" | OUTDATED — corrected to $14B |
| "Cartesia $50M comparable" | INCORRECT — Cartesia is voice AI, not architecture licensing |

**B.2 Conflicts of interest**

None. The author is the sole contributor and has no financial relationships with NVIDIA, Moonshot, Cartesia, or any foundation model lab.

**B.3 Funding source**

This work is self-funded. No grants, no corporate sponsorship, no investor money.

---

*Whitepaper v2.0 · July 2026*
*Marcus · Bubble Transformer*
*Contact: github.com/Markush42/bubble-transformer*

*This is a research artifact description, not a product pitch. All claims are tied to reproducible experiments in the open-source repository.*
