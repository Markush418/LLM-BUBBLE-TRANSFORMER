# Bubble Transformer V5: How We Broke Through the 2% PPL Gate

**Marcus | July 2026**

---

## TL;DR

After 38 failed configurations, we finally passed the BT V5 perplexity gate. FocusDeltaNet achieves **+0.16% PPL** on Qwen3-0.6B (22.550 vs 22.513 baseline) — 12.5x better than the 2% threshold. The secret? Use Sinkhorn for token grouping, not attention normalization.

---

## The Problem

Previous BT V5 variants tried replacing softmax with doubly-stochastic attention (SIRI). All failed the ≤2% PPL gate:

- Pure SIRI: +211% PPL (destroyed peakedness)
- Row-stochastic: +30% PPL
- Hybrid approaches: +2.39% to +2.85% PPL

The core issue: **doubly-stochastic normalization destroys the sharp attention distributions language models need.**

---

## The Insight

The Focus paper (arXiv:2604.03260) showed that Sinkhorn can *improve* PPL when used for **token grouping** rather than attention normalization.

Key idea:
1. Use Sinkhorn to create soft group assignments
2. Apply standard softmax **within** each group
3. Preserve peakedness while adding geometric structure

Focus achieved 29% improvement on GPT-2 with this approach.

---

## Our Architecture

**FocusBubbleAttention** pipeline:

```
S = Q @ K^T / sqrt(d)           # dot-product scores
S = S + psi                      # Power Diagram bias
groups = Sinkhorn(S, tau=1)      # soft grouping (doubly-stochastic)
attn = softmax(S + log(groups))  # softmax within groups (preserved peakedness)
output = attn @ V
```

**FocusDeltaNet** adds DeltaNet delta rule:

```
out_delta = DeltaNet(Q, K, V)      # linear O(N)
out_focus = FocusBubble(Q, K, V)   # softmax within groups
output = λ * out_delta + (1-λ) * out_focus
```

---

## Results

### Single Layer (eps=0.001, tau=1)

| Layer | PPL | Δ% | Gate |
|-------|-----|-----|------|
| L7 | 22.681 | +0.74% | PASS |
| L12 | 22.706 | +0.86% | PASS |
| L10 | 22.757 | +1.08% | PASS |

### FocusDeltaNet (lambda sweep at L7)

| λ | PPL | Δ% | Gate |
|---|-----|-----|------|
| 0.0 | 22.681 | +0.74% | PASS |
| 0.3 | **22.550** | **+0.16%** | PASS |
| 0.5 | 22.651 | +0.61% | PASS |
| 1.0 | 23.558 | +4.64% | FAIL |

**Optimal: λ=0.3 (30% DeltaNet, 70% Focus)**

### Multi-Layer Scaling

| Config | PPL | Δ% | Gate |
|--------|-----|-----|------|
| FocusDeltaNet L7 | 22.550 | +0.16% | PASS |
| FocusDeltaNet L7+L10 | 22.648 | +0.60% | PASS |
| FocusDeltaNet L7+L10+L12 | 22.825 | +1.38% | PASS |

**Key finding: FocusDeltaNet scales to 3 layers** while pure Focus degrades.

### Long-Context (NIAH at 2K tokens)

All configurations achieve **100% retrieval accuracy**. Focus Bubble preserves sharp attention patterns needed for needle retrieval.

---

## The L9 Anomaly

L9 fails the gate (+2.28%) while neighbors L10/L11/L12 pass (+1.08%, +1.15%, +0.86%). Per-head analysis shows L9's outlier heads (H2: 131%, H6: 154%) are not geometric outliers compared to L10.

**Root cause unknown** — appears to be integration effects across heads, not individual head geometry.

---

## Honest Caveats

1. **Single-layer swap only** — not validated with pretrain from scratch
2. **Limited long-context** — NIAH at 2K tokens (GTX 1650 VRAM limit)
3. **No downstream evaluation** — MMLU, HumanEval, GSM8K not tested
4. **Power Diagram ψ absorbed** — no effect on output (absorbed by Sinkhorn column normalization)
5. **L9 anomaly unexplained** — root cause not identified

---

## What's Next

1. **Pretrain Bubble-1.3B** from scratch (needs H100/A100)
2. **Long-context evaluation** at 4K-32K tokens
3. **Downstream task evaluation** (MMLU, HumanEval, GSM8K)
4. **Speed optimization** (CUDA kernel for Focus Bubble)

---

## Reproducibility

- **Code**: https://github.com/Markush418/LLM-BUBBLE-TRANSFORMER
- **Tests**: 501 passing, 0 failed
- **Hardware**: GTX 1650 (4.3GB VRAM)
- **Time**: ~30 minutes for full benchmark suite

```bash
# Run the benchmarks
py experiments/benchmark_focus_fine_sweep.py
py experiments/benchmark_focus_layer_sweep_optimal.py
py experiments/benchmark_focus_deltanet_sweep.py
py experiments/benchmark_focus_multilayer.py
```

---

## Conclusion

Focus Bubble V5 passes the BT V5 perplexity gate by using Sinkhorn for token grouping and preserving softmax within groups. The optimal configuration (L7, λ=0.3) achieves +0.16% PPL — 12.5x better than the threshold.

This validates the hypothesis that SIRI's failure to preserve softmax peakedness can be overcome by restricting doubly-stochastic normalization to a grouping role rather than replacing attention entirely.

**We're seeking funding to test whether this survives end-to-end pretrain and long-context evaluation.**

---

*Bubble Transformer V5 | July 2026*
*Status: Research artifact, not production system*
*arXiv: Coming soon*
