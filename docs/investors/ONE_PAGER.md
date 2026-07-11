# Bubble Transformer

**Research-stage hybrid attention for post-softmax architectures**

---

## What This Is (Honest Framing)

Bubble Transformer V4 is an **experimental hybrid attention mechanism** combining three research directions:
1. **DeltaNet** (NeurIPS 2024) — linear $O(N)$ recurrence via the delta rule
2. **SIRI/Sinkhorn post-processing** (Sinkformers, AISTATS 2022) — doubly-stochastic regularization
3. **Learnable per-query scalar bias ψ** (our contribution) — geometric structure injection

**Current status**: validated as a drop-in wrapper on Qwen3-0.6B with WikiText-2. **Not trained from scratch. Not deployed in production. Not benchmarked on long-context or downstream tasks.**

This document does not claim production readiness. It claims a reproducible research result worth $800k of seed funding to validate end-to-end.

---

## The Research Finding Worth Funding

**Discovery**: raw SIRI (doubly-stochastic attention) catastrophically destroys softmax peakedness in language modeling. Our benchmarks on Qwen3-0.6B:

| Configuration | WikiText-2 PPL | Δ vs baseline |
|---|---|---|
| Baseline (softmax, all 28 layers) | 22.515 | — |
| Pure SIRI (L3 only, λ=0.0) | 30.14 | +7.6 |
| Soft blend SIRI+softmax (L3, α=0.7) | 26.76 | +4.2 |
| Pure ΔNet (L3 only, λ=1.0) | **23.749** | **+1.24** |

**The +545 PPL claim from the previous version was misleading** — that came from swapping 4 mid-layers in a frozen model, which would break any attention mechanism. The honest data point is single-layer: SIRI adds +7.6 PPL, and our soft blend recovers 45% of the gap.

**Best result**: pure DeltaNet at single layer, +1.24 PPL. The SIRI component does not help language modeling in our frozen-swap setup. The geometric bias ψ is unvalidated (zero-initialized, contributes zero PPL drift).

**What this means**: we have a **DeltaNet wrapper with strong numerical safety** (462 tests passing, 6 documented bug fixes) and a **research hypothesis** (soft blend may help in full pretrain) worth testing.

---

## What We Are NOT Claiming

To preempt investor due diligence:

- ❌ **Not competitive with Kimi Linear** (arXiv:2510.26692). They trained 3B/48B params and beat full MLA. We have 0.6B frozen swap.
- ❌ **Not competitive with Gated DeltaNet** (ICLR 2025, NVIDIA). They have NVIDIA-validated training, FLA library integration, and brand.
- ❌ **Not competitive with Mamba-3** (ICLR 2026 Oral). They have state-of-the-art efficiency and Together AI production deployment.
- ❌ **Patent not filed**. The provisional application referenced in earlier drafts was an overstatement. We are evaluating patent strategy pending pretrain validation.
- ❌ **Not published at NeurIPS 2026**. We are preparing an arXiv preprint for Q3 2026.
- ❌ **No long-context validation**. All benchmarks are 256 tokens. Kimi Linear targets 128k.
- ❌ **No downstream task evaluation**. We measure PPL, not MMLU/HumanEval/GSM8K.

---

## What We Are Claiming (With Evidence)

1. **Reproducible research artifact**: 462 tests passing, 0 failed. One-command benchmark: `python experiments/run_experiment.py --mode real`. GitHub public.

2. **Engineering depth in numerical safety**: 6 documented bugs found and fixed during development (causal mask shape, SIRI saturation, dtype downcast, test isolation, chiller row sums, sparse mask leakage). This is the kind of work that takes competitors 6+ months to replicate.

3. **A research hypothesis worth $800k to test**: Does soft blend (softmax + SIRI) help in end-to-end pretrain? Frozen-swap suggests +1.24 PPL is achievable with pure ΔNet. Pretrain may reveal different trade-offs.

4. **A defensible engineering wrapper**: Drop-in replacement for any HuggingFace transformer with `attn_implementation="eager"`. Works on Qwen3-0.6B with GQA, KV cache, and 4-bit quantization.

5. **Open-source commitment**: Apache 2.0 for research. Commercial licensing available for foundation model training.

---

## Market Context (Without Inflation)

The linear attention space is **crowded and validated**:

- **Mamba-3** (ICLR 2026 Oral, Cartesia $191M raised, Together AI deployment): SSM-based, production-ready
- **RWKV-7** (Mar 2025, 2.9B params SoTA on multilingual): linear RNN, open-source
- **Gated DeltaNet** (ICLR 2025, NVIDIA Labs, FLA library): delta rule + gating, NVIDIA-validated
- **Kimi Linear** (Oct 2025, Moonshot AI): KDA + MLA hybrid, 3B/48B trained, beat full MLA

**Bubble's positioning**: we are NOT competing on architecture novelty. Mamba/Kimi/Gated DeltaNet already won that race. We are exploring the **geometric bias ψ** axis — a small, orthogonal contribution that may provide interpretability benefits independent of PPL.

**Realistic market entry**: not "disrupt $200B foundation model training market." More like "contribute a small, defensible building block to the linear attention ecosystem" with potential acqui-hire by NVIDIA Labs, Cartesia, or Moonshot if ψ proves useful in pretrain.

---

## Comparable Exits (Corrected)

Previous version listed Mistral at $7B and Cartesia at $50M. Corrected:

- **Mistral AI**: $14B valuation (Sep 2025, ASML-led $2B Series C), $100M ARR. They build foundation models, not license architecture.
- **Cartesia**: $191M raised (Series A 2025). They sell voice AI products, not architecture licensing. Not a direct comparable.
- **RWKV**: community-driven, no clear valuation. Open-source foundation with Bo Peng as BDFL.

**None of these are "architecture licensing" precedents**. Our licensing thesis is speculative. The more likely outcome is acqui-hire ($5-30M) or independent growth to $5-10M ARR via consulting + custom training.

---

## Business Model (Realistic)

**Phase 1 (Year 1)** — $100k–$500k ARR
Custom attention integration consulting for AI labs. We help teams add Bubble-style wrappers to their training pipelines. Target: 2-3 design partners.

**Phase 2 (Year 2)** — $500k–$2M ARR
License the wrapper to enterprises training custom LLMs. Per-engineer pricing ($50k-200k/year). Target: NVIDIA, Cohere, AI21, Together AI, Anyscale.

**Phase 3 (Year 3+)** — $2M–$10M ARR
Either (a) successful licensing with multiple enterprise customers, or (b) acqui-hire by a foundation model lab.

**Exit scenarios** (honest):
1. Acqui-hire by NVIDIA Labs / Cartesia / Moonshot for $5M–$30M (most likely if ψ proves useful in pretrain)
2. Independent growth to $10M ARR with Series A in Year 3 (less likely — competitive space)
3. Open-source the ψ work and pivot to consulting (safe baseline)

---

## Technical Roadmap (Realistic 12 Months)

| Quarter | Milestone | Cost | Risk |
|---|---|---|---|
| Q1 | Bubble-1.3B from scratch, 50B tokens, ΔNet wrapper | $200k compute | High — may not beat Qwen2.5-1.5B |
| Q2 | Ablation: ψ enabled vs ψ=0 in pretrain | $100k compute | Medium — ψ may not help |
| Q3 | Long-context NIAH at 4k-8k vs Gated DeltaNet | $50k | Medium — long-context is hard |
| Q4 | Downstream eval: HellaSwag, LAMBADA, ARC | $50k | Low — standard evals |

**Total compute budget**: $400k. Requires partnership with Lambda Labs, RunPod, or CoreWeave for discounted rates.

**If Q1 milestone fails** (Bubble-1.3B underperforms Qwen2.5-1.5B by >5 PPL): pivot to (a) ψ-focused interpretability research, (b) consulting revenue, or (c) acqui-hire.

---

## Team (Honest)

**Marcus** (Founder, AI Researcher)
- Designed and implemented Bubble V4 from scratch
- 462-test suite, 6 documented bug fixes, open-source core
- Background: AI infrastructure, mathematical optimization, PyTorch
- **No PhD, no published papers, no prior startup exit**

**Current gap**: need a CTO with large-scale pretrain experience. $800k budget includes $200k for CTO hire (1 year, including equity).

**Seeking**: senior ML engineer with pretrain experience (e.g., ex-Mistral, ex-Cartesia, ex-NVIDIA) or PhD researcher in efficient attention.

---

## Why Fund This (If You Believe the Thesis)

The investment thesis is **not** "Bubble will beat Mamba-3." That ship has sailed.

The thesis is:

1. **The linear attention space is large and fragmented**. Mamba, RWKV, Gated DeltaNet, Kimi all have different strengths. There is room for a focused contribution on the geometric bias axis.

2. **Interpretability is an underserved orthogonal axis**. Every linear attention architecture optimizes for PPL and throughput. None optimize for "can we understand what this head is computing?" If ψ provides interpretable structure, that is independently valuable to Anthropic, DeepMind, and MIT's interpretability teams.

3. **First-mover in "geometric doubly-stochastic attention" (GDSA)**. No published paper combines doubly-stochastic regularization with learnable geometric bias. A clean arXiv preprint on this would establish the field and provide citation advantage.

4. **The engineering wrapper is real value**. 462 tests, GQA support, KV cache compatibility, dtype handling — this is non-trivial to replicate. A foundation model lab could save 3-6 engineer-months by licensing.

**Risk**: 80% chance pretrain shows that ψ provides no benefit over ΔNet, and the wrapper is just an engineering product. In that case, the company is a consulting shop, not a $100M ARR licensing business.

---

## Ask

**$800k seed** for 12-month roadmap:
- 50% compute ($400k): Bubble-1.3B pretrain, ablations, long-context eval
- 25% engineering ($200k): CTO hire, 1 senior ML engineer
- 15% research ($120k): PhD intern, conference submissions
- 10% GTM ($80k): 1 BD for design partner conversations

**Use of funds is honest**: $400k compute gets you 2000 H100-hours. That is sufficient for a 1.3B model / 50B tokens, comparable to DeltaNet paper's training setup. Not sufficient for 7B+.

**Dilution**: 15-20% for $800k seed (post-money $4-5.3M).

**Milestone for Series A** (target Q4 2026): Bubble-1.3B trained, ψ ablation completed, 2 design partner LOIs, $200k ARR from consulting.

---

## Contact

**Marcus · Bubble Transformer**
GitHub: [github.com/Markush42/bubble-transformer](https://github.com/Markush42/bubble-transformer)
Demo: `python experiments/run_experiment.py --mode real` on any GPU with 4GB+ VRAM

*This is a pre-revenue, research-stage opportunity. 80% chance of moderate outcome, 15% chance of acqui-hire, 5% chance of independent growth to $10M ARR. Expected return: 2-5x over 3 years if pretrain validates ψ.*

---

*Last updated: July 2026*
*This document supersedes all previous drafts. Numbers are honest, not aspirational.*
