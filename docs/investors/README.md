# Investor Documentation (v2.0 — Honest)

This folder contains materials for institutional investors, VCs, and strategic partners evaluating Bubble Transformer for funding, licensing, or acqui-hire.

---

## ⚠️ What Changed in v2.0

After red-team review, we identified and corrected several overstatements in v1.0:

| v1.0 Claim | v2.0 Correction |
|---|---|
| "Soft blend preserves both properties" | **Removed** — destroys doubly-stochasticity |
| "First architecture to integrate Power Diagrams" | **Corrected** — ψ is a scalar bias, not a true Power Diagram |
| "Patent pending (USPTO #62/XXX,XXX)" | **Removed** — no filing |
| "NeurIPS 2026 submission" | **Removed** — no submission |
| "Validated in production" | **Corrected** to "research artifact" |
| "Mistral $7B valuation" | **Updated** to $14B |
| "Cartesia $50M comparable" | **Corrected** — different business model |
| "+545 PPL" as headline | **Corrected** to +7.6 PPL (single-layer, honest) |

**The core thesis remains**: we built a DeltaNet wrapper with strong numerical safety, and we want $800k to test whether our soft blend + geometric bias hypothesis survives end-to-end pretrain. We are not claiming to beat Mamba-3, Gated DeltaNet, or Kimi Linear.

---

## Documents

### [ONE_PAGER.md](./ONE_PAGER.md)
**Pitch deck one-pager** (2-3 pages, ~5 min read)

For first meetings with VCs. Covers:
- Honest framing (what we are NOT claiming)
- The research finding worth funding (single-layer +1.24 PPL)
- Market context (where Bubble fits in the crowded linear attention space)
- Realistic business model (consulting → licensing → acqui-hire)
- 12-month roadmap with explicit failure criteria

**Use this for**: initial outreach, first pitch meeting, cold emails.

### [WHITEPAPER.md](./WHITEPAPER.md)
**Technical whitepaper** (15-20 pages, ~30 min read)

For technical due diligence. Covers:
- Mathematical foundations with honest caveats
- Single-layer frozen swap experiments (not pretrain)
- The SIRI peakedness problem and partial soft blend solution
- Geometric bias ψ: an unvalidated hypothesis
- 6 documented numerical bugs and fixes
- 462-test suite description
- Explicit limitations and failure modes
- Comparison with Mamba-3, Gated DeltaNet, Kimi Linear
- Open research questions
- Corrected mathematical claims (Appendix A)
- Honest disclosures (Appendix B)

**Use this for**: technical partner review, scientific advisory board, patent strategy, arXiv preprint preparation.

---

## Key Talking Points (v2.0)

If you only have 2 minutes, lead with these three points:

1. **"We built a DeltaNet wrapper with 462 tests and +1.24 PPL single-layer swap on Qwen3-0.6B. We're not competing with Mamba-3 or Kimi Linear — they're 18 months ahead of us."**

2. **"We have a research hypothesis (soft blend + learnable geometric bias ψ) that may or may not survive end-to-end pretrain. $800k is to test it honestly."**

3. **"The most likely outcome is acqui-hire ($5-30M) by NVIDIA Labs, Cartesia, or Moonshot if ψ proves useful. Independent growth to $10M ARR is possible but less likely. We're not promising $100M ARR."**

---

## What Investors Will Ask (and How We Answer)

### Q: "Why not just use Mamba-3?"

**A**: Mamba-3 is SSM-based, not delta-rule. We are not competing on architecture novelty. Our exploration is orthogonal: whether a learnable geometric bias provides interpretability value in linear attention contexts. Mamba-3 doesn't have this axis. We are also 18 months behind in funding and team size, which is honest.

### Q: "Your single-layer PPL doesn't matter without pretrain."

**A**: Correct. That's why $200k of the $800k is compute for Bubble-1.3B from scratch. If that milestone fails (PPL gap > 5 vs Qwen2.5-1.5B), we pivot to consulting or acqui-hire.

### Q: "Kimi Linear already beat full MLA. What's your edge?"

**A**: Kimi Linear is KDA + MLA hybrid, full pretrain, $100M+ compute budget. We are not competing with them. Our edge is the engineering wrapper (GQA, KV cache, dtype safety) and the research hypothesis about ψ.

### Q: "Your team is 1 founder. Why $800k?"

**A**: $200k of the $800k is for a CTO hire with pretrain experience. Without CTO, no pretrain. Without pretrain, no validation. $800k is the minimum to get to Bubble-1.3B trained + 1 design partner LOI.

### Q: "What if ψ doesn't help?"

**A**: That's the bet. 80% chance pretrain shows ψ provides no benefit, and we pivot to (a) consulting revenue from the engineering wrapper, or (b) acqui-hire for $5-30M. 20% chance ψ provides interpretability or generalization value, leading to Series A and independent growth.

---

## Validation

All technical claims in these documents are backed by:
- 462 passing tests in the open-source repository
- Reproducible benchmarks with one command: `python experiments/run_experiment.py --mode real`
- Public GitHub: [github.com/Markush42/bubble-transformer](https://github.com/Markush42/bubble-transformer)

**What the benchmarks do NOT show**:
- Pretrain validation (we never trained a model with Bubble)
- Long-context performance (256 tokens only)
- Downstream task accuracy (PPL only, not MMLU/HumanEval/GSM8K)
- Multi-architecture support (Qwen3 only)

---

## Red Flags We Acknowledge

1. **Team**: 1 founder, no PhD, no published papers, no prior startup exit
2. **Track record**: zero revenue, zero users, zero production deployment
3. **Competition**: Mamba-3, Gated DeltaNet, Kimi Linear are 18+ months ahead
4. **Claim quality**: previous v1.0 had overstated claims; v2.0 is corrected
5. **Exit path**: most likely outcome is acqui-hire, not $100M ARR

---

## Contact

**Marcus · Bubble Transformer**
GitHub: [github.com/Markush42/bubble-transformer](https://github.com/Markush42/bubble-transformer)

*Pre-revenue, pre-Series A, pre-pretrain. Seeking $800k seed for 12-month roadmap to Bubble-1.3B + design partner validation.*

*Honest expected return: 2-5x over 3 years if pretrain validates ψ; 0.5-1x (acqui-hire) if it doesn't.*
