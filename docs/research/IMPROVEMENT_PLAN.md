# Research-Driven Improvement Plan for Bubble Transformer

**Based on arXiv literature review (2024-2026) — 34 papers analyzed**

**Date**: July 2026
**Status**: Action plan for v5.0 architecture

---

## Executive Summary

After reviewing 34 recent arXiv papers across four relevant research areas (long-context evaluation, doubly-stochastic attention, geometric structure in attention, and hybrid architectures), we identified **8 concrete improvements** that would strengthen Bubble Transformer's scientific rigor, competitive position, and investor narrative.

The single most important finding: **the SIRI peakedness problem has known partial solutions in the literature** (Sparse Sinkhorn, Gumbel-Sinkhorn, Sliced Transport) that we should test before claiming soft blend as our solution.

---

## Gap 1: No Long-Context Evaluation

### Current State
All Bubble V4 benchmarks are 256 tokens. We have no evidence that the architecture works in the long-context regime where linear attention is most valuable.

### Evidence from Literature
- **RULER** (Hsieh et al., NVIDIA, [2404.06654](https://arxiv.org/abs/2404.06654)): Models advertising 32K+ often collapse at 32K. Without RULER-style evaluation, our 256-token benchmark is meaningless for the long-context thesis.
- **BABILong** ([2406.10149](https://arxiv.org/abs/2406.10149)): Popular LLMs use only 10-20% of nominal context. This is the bullseye target for SIRI's doubly-stochastic normalization.
- **LongBench v2** ([2412.15204](https://arxiv.org/abs/2412.15204)): 503 multiple-choice questions, 8K-2M words. Code repo and structured data are natural targets for geometric partitioning.
- **MECW** (Paulsen, [2509.21361](https://arxiv.org/abs/2509.21361)): MECW is up to 99% smaller than nominal context. The 99% gap is exactly what Bubble should target.

### Recommended Action
| Priority | Task | Timeline |
|---|---|---|
| **P0** | Implement NIAH-1 at 4K, 8K, 16K, 32K context | 1 week |
| **P0** | Run RULER 13-task suite on Qwen3-0.6B + Bubble wrapper | 2 weeks |
| **P1** | BABILong reasoning tasks at 10K-50K context | 2 weeks |
| **P1** | LongBench v2 multi-doc QA subset | 1 week |

### Files to Create/Modify
- `experiments/long_context/niah_bubble.py` — NIAH benchmark
- `experiments/long_context/ruler_bubble.py` — RULER adapter
- `tests/test_long_context.py` — validation tests
- `docs/results/long_context_report.md` — results

---

## Gap 2: SIRI Peakedness Problem Has Known Solutions

### Current State
We claim "soft blend" (α=0.7) as our partial solution to SIRI destroying peakedness (+7.6 PPL). The literature contains **5+ alternative approaches** that we have not tested.

### Evidence from Literature
1. **Sparse Sinkhorn Attention** (Tay et al., ICLR 2020, [2002.11296](https://arxiv.org/abs/2002.11296)): Apply top-k sparse attention first, then Sinkhorn-balance the residual. Directly addresses peakedness loss.

2. **Gumbel-Sinkhorn Networks** (Mena et al., ICLR 2018, [1802.08665](https://arxiv.org/abs/1802.08665)): Gumbel noise + Sinkhorn → near-permutation matrix. Inherently preserves peakedness due to sparsification.

3. **ESPFormer** (Shahbazi et al., [2502.07962](https://arxiv.org/abs/2502.07962)): Sliced optimal transport replaces iterative Sinkhorn. Produces DSMs that are "more diverse" and preserve more information.

4. **LOTFormer** (Shahbazi et al., [2509.23436](https://arxiv.org/abs/2509.23436)): Low-rank OT via pivot measure. Pivots act as "anchors" that capture dominant modes before balance.

5. **Sinkhorn Rank Decay Analysis** (Lapenna et al., [2604.07925](https://arxiv.org/abs/2604.07925)): Theoretical bound: rank decays doubly-exponentially with depth under pure Sinkhorn. Skip connections are non-negotiable.

### Recommended Action
| Priority | Task | Timeline |
|---|---|---|
| **P0** | Implement Gumbel-Sinkhorn attention, benchmark on WikiText-2 | 1 week |
| **P0** | Implement Sparse+Sinkhorn (top-k first), benchmark | 1 week |
| **P1** | Implement ESPFormer sliced transport, benchmark | 2 weeks |
| **P2** | Test LOTFormer low-rank pivots | 2 weeks |

### Files to Create/Modify
- `experiments/attention_variants/gumbel_sinkhorn.py` — Gumbel-Sinkhorn implementation
- `experiments/attention_variants/sparse_sinkhorn.py` — Sparse-then-Sinkhorn
- `experiments/attention_variants/espformer.py` — Sliced transport
- `experiments/compare_attention_variants.py` — head-to-head benchmark
- `tests/test_attention_variants.py` — unit tests

### Expected Impact
If any of these matches or beats the +1.24 PPL of pure ΔNet, we have a **stronger soft blend story** (or replace soft blend entirely). Either way, this converts "we discovered peakedness loss" into "we tested 5 solutions and found the best one."

---

## Gap 3: Geometric Bias ψ Lacks Theoretical Grounding

### Current State
The geometric bias ψ is described as a "learnable per-query scalar bias" with no formal connection to Power Diagrams or other geometric structures. This weakens our intellectual positioning.

### Evidence from Literature
- **Expressivity of Transformers: A Tropical Geometry Perspective** ([2604.14727](https://arxiv.org/abs/2604.14727)): Self-attention evaluates exactly to a **Power Voronoi Diagram** in the temperature-zero limit. This is the formal connection we need.

- **Chessformer Geometric Attention Bias** ([2605.19091](https://arxiv.org/abs/2605.19091)): Square-token design enables attention patterns directly attributable to board squares. GAB adds geometric inductive bias.

- **Accelerating Robot Path Planning via Voronoi diagrams** ([2605.28362](https://arxiv.org/abs/2605.28362)): Voronoi-based deformable attention with topological continuity loss.

- **Texture-Shape Voronoi Style Diversification** ([2606.15072](https://arxiv.org/abs/2606.15072)): Voronoi-style diversification modifies textures preserving scene geometry.

### Key Insight
The tropical geometry paper directly proves that **MHSA is a Power Voronoi Diagram in the limit**. This means our ψ bias is mathematically equivalent to weighting the Power Diagram cells. We can frame ψ as a **learnable per-cell weight** in the formal Power Voronoi construction, not just a "scalar bias."

### Recommended Action
| Priority | Task | Timeline |
|---|---|---|
| **P0** | Rewrite §5 of WHITEPAPER to cite [2604.14727] and formalize ψ as Power Voronoi cell weight | 3 days |
| **P1** | Implement per-cell-weight variant of ψ (bilinear form instead of linear) | 1 week |
| **P1** | Test ψ on synthetic Voronoi-partition tasks to verify it learns cell structure | 2 weeks |
| **P2** | Visualize learned ψ for a 2D toy problem to show geometric structure | 1 week |

### Files to Create/Modify
- `experiments/power_voronoi/pv_attention.py` — formal Power Voronoi attention
- `experiments/power_voronoi/toy_voronoi.py` — 2D toy problem
- `docs/decisions/2026-07-05-power-voronoi-formalization.md` — design doc

---

## Gap 4: No Hybrid Architecture Comparison

### Current State
Bubble V4 is positioned as a standalone attention mechanism. The literature shows that **hybrid linear+softmax is the current best practice** (Jamba, Zamba, Striped Hyena, Hymba, HOLA).

### Evidence from Literature
- **HOLA** (Cui, [2607.02303](https://arxiv.org/abs/2607.02303)): Combines delta-rule (compressive memory) with bounded exact KV cache (hipocampal complement). 340M params, 15B tokens → WikiText PPL 27.32 → 22.92 (-16.1%). **Robust in RULER to 32k (16x training length)**.
- **Jamba** (AI21): 52B total / 12B active, 8:1 Mamba:Transformer ratio. Production-deployed.
- **CARVE** (Dutta, [2606.27229](https://arxiv.org/abs/2606.27229)): Corrects 3 GDN-2 defects. 1.3B params, 100B tokens → WikiText PPL 15.72. SOTA in 9 common-sense + all RULER probes.
- **Erase-then-Delta Attention** (Alibaba, [2606.26560](https://arxiv.org/abs/2606.26560)): Decouples erase/write addresses. 2.5B dense + 25B-A2.8B MoE tested. 4K-128K context.
- **FlashMorph** (Lan et al., [2606.30562](https://arxiv.org/abs/2606.30562)): Training-free hybrid layer selection via budget-constrained optimization.

### Recommended Action
| Priority | Task | Timeline |
|---|---|---|
| **P0** | Implement Bubble-Softmax hybrid (N layers Bubble, M layers softmax) | 1 week |
| **P0** | Compare against HOLA design (delta + exact KV) on WikiText-2 | 1 week |
| **P1** | Add layer selection heuristic: NLL-guided (like [2606.27791](https://arxiv.org/abs/2606.27791)) | 2 weeks |
| **P2** | Implement alternating pattern (Mamba-style: every Nth layer is full attention) | 1 week |

### Files to Create/Modify
- `experiments/hybrid/bubble_softmax_hybrid.py` — hybrid layer wrapper
- `experiments/hybrid/layer_selection.py` — NLL-guided selection
- `experiments/hybrid/compare_hybrids.py` — benchmark all hybrid configs
- `tests/test_hybrid.py` — unit tests

### Expected Impact
Hybrid architectures are the SOTA. By testing Bubble as a **component** in a hybrid stack (rather than a standalone replacement), we align with industry practice and gain competitive relevance.

---

## Gap 5: No Mechanistic Interpretability Story

### Current State
We mention "interpretability via ψ bias" in the WHITEPAPER but provide no concrete interpretability analysis.

### Evidence from Literature
- **Emergent Capabilities from Sparse Attention** ([2606.25010](https://arxiv.org/abs/2606.25010)): Capabilities emerge from learning difficulty of sparse attention patterns. More heads improves learning efficiency. **Direct mechanistic insight** we can apply.
- **Explaining RhythmFormer** ([2606.13839](https://arxiv.org/abs/2606.13839)): Beyond Intuition method reaches refined skin coverage 0.83, faithfulness F=0.92. Methodology adaptable to Bubble.
- **SpotAttention** ([2606.22874](https://arxiv.org/abs/2606.22874)): KL-distilled selector estimates attention distribution. Could be applied to learn **which heads use SIRI vs ΔNet in soft blend**.
- **Chessformer** ([2605.19091](https://arxiv.org/abs/2605.19091)): Square-tokens enable direct attribution of attention to board squares. **Analogue for Bubble**: we can use Power Voronoi cells to attribute attention to geometric regions.

### Recommended Action
| Priority | Task | Timeline |
|---|---|---|
| **P0** | Train a 2D toy model with Power Voronoi attention, visualize cells | 1 week |
| **P0** | Run probing analysis: which attention heads in Qwen3-0.6B use SIRI vs ΔNet in soft blend? | 2 weeks |
| **P1** | Implement Beyond Intuition-style attribution for Bubble attention | 2 weeks |
| **P2** | Generate "attention map" visualization for Bubble vs softmax on 5 sample inputs | 1 week |

### Files to Create/Modify
- `experiments/interpretability/toy_voronoi_viz.py` — 2D toy with viz
- `experiments/interpretability/probe_siri_deltanet.py` — head-level analysis
- `experiments/interpretability/beyond_intuition.py` — attribution method
- `docs/results/interpretability_report.md` — findings

---

## Gap 6: No Throughput / Cost Benchmark

### Current State
We claim "$O(N)$ cost advantage" without measuring actual throughput. The cost story is a critical investor talking point.

### Evidence from Literature
- **HYPIC** (Liu et al., [2607.01299](https://arxiv.org/abs/2607.01299)): First serving system for hybrid-attention LLMs. TTFT reduction 2.45x, peak throughput +2.0x. Methodology adaptable.
- **Nemotron-Labs-TwoTower** ([2606.26493](https://arxiv.org/abs/2606.26493)): 2.42x throughput wall-clock vs AR baseline. Two-tower architecture.
- **SpotAttention** ([2606.22874](https://arxiv.org/abs/2606.22874)): 3.9x faster decode than FlashAttention, 1.8x vs Twilight. INT4/FP4 quantization reduces cache 3.5x.

### Recommended Action
| Priority | Task | Timeline |
|---|---|---|
| **P0** | Benchmark tokens/sec for Bubble vs softmax on Qwen3-0.6B at 256, 1K, 4K context | 3 days |
| **P0** | Measure peak memory and KV cache size | 3 days |
| **P1** | Compare with SpotAttention INT4/FP4 quantization | 1 week |
| **P2** | Add kernel optimization (Triton/CUDA) for SIRI step | 4 weeks |

### Files to Create/Modify
- `experiments/throughput/benchmark_throughput.py` — tokens/sec
- `experiments/throughput/benchmark_memory.py` — peak memory
- `experiments/throughput/quantize_int4.py` — INT4/FP4 support
- `docs/results/throughput_report.md` — results

---

## Gap 7: No Downstream Task Evaluation

### Current State
All benchmarks are PPL on WikiText-2. We have not evaluated on MMLU, HellaSwag, GSM8K, HumanEval, or any downstream task.

### Evidence from Literature
- **HOLA** ([2607.02303](https://arxiv.org/abs/2607.02303)): Robust in RULER to 32k context. Multi-task evaluation is standard.
- **CARVE** ([2606.27229](https://arxiv.org/abs/2606.27229)): SOTA in 9 common-sense benchmarks. All RULER probes.
- **EDA** ([2606.26560](https://arxiv.org/abs/2606.26560)): Persists after 80B-token long-context midtraining. Multi-scale evaluation.

### Recommended Action
| Priority | Task | Timeline |
|---|---|---|
| **P0** | Run HellaSwag, ARC-Easy, LAMBADA on Qwen3-0.6B + Bubble wrapper | 1 week |
| **P0** | Run MMLU (5-shot) | 1 week |
| **P1** | Run GSM8K, HumanEval | 2 weeks |
| **P2** | Long-context: LongBench v2 subset | 1 week |

### Files to Create/Modify
- `experiments/downstream/hellaswag.py` — HellaSwag eval
- `experiments/downstream/mmlu.py` — MMLU eval
- `experiments/downstream/gsm8k.py` — GSM8K eval
- `docs/results/downstream_report.md` — results

---

## Gap 8: WHITEPAPER Positioning vs Literature

### Current State
WHITEPAPER v2.0 claims "we are not competing with Mamba-3, Gated DeltaNet, Kimi Linear." This is honest but defeats the differentiation story.

### Evidence from Literature
Multiple 2026 papers show **incremental improvements** on top of existing architectures (CARVE on GDN-2, EDA on DeltaNet, FlashMorph on hybrid). Bubble can position as **incremental improvement on DeltaNet** with specific advantages:

1. **Geometric inductive bias** via ψ (formalized as Power Voronoi cell weight)
2. **Plausibly better long-context** via doubly-stochastic normalization (BABILong hypothesis)
3. **Interpretability story** via Power Voronoi cell visualization

### Recommended Action
| Priority | Task | Timeline |
|---|---|---|
| **P0** | Reframe WHITEPAPER §1.1 as "incremental improvement on DeltaNet" not "competitor to Mamba-3" | 1 day |
| **P0** | Reframe §3.4 (Power Diagram) using tropical geometry citation [2604.14727] | 1 day |
| **P1** | Add §6.6 direct comparison to Gated DeltaNet on WikiText-2 | 3 days |
| **P1** | Add §6.7 throughput comparison to Mamba-3 | 1 week |
| **P2** | Rewrite §10 (Limitations) to acknowledge 2026 SOTA and our incremental position | 1 day |

### Files to Create/Modify
- `docs/investors/WHITEPAPER.md` — sections 1, 3, 6, 10
- `docs/investors/ONE_PAGER.md` — thesis statement

---

## Summary: Priority-Ordered Action Plan

| # | Gap | Impact | Effort | Priority |
|---|---|---|---|---|
| 1 | No long-context evaluation | HIGH | 2 weeks | **P0** |
| 2 | SIRI peakedness has known solutions | HIGH | 2 weeks | **P0** |
| 3 | Geometric bias ψ lacks theoretical grounding | MEDIUM | 3 days | **P0** |
| 4 | No hybrid architecture comparison | HIGH | 2 weeks | **P0** |
| 5 | No mechanistic interpretability story | MEDIUM | 3 weeks | **P1** |
| 6 | No throughput / cost benchmark | HIGH | 3 days | **P0** |
| 7 | No downstream task evaluation | HIGH | 2 weeks | **P0** |
| 8 | WHITEPAPER positioning vs literature | MEDIUM | 1 week | **P0** |

**Total P0 effort**: ~8 weeks of focused work
**Total P1+P2 effort**: ~6 additional weeks

**Recommendation**: prioritize Gaps 2, 3, 6, 8 (2-3 weeks) for the next investor pitch. Gaps 1, 4, 7 (6-8 weeks) for the next arXiv preprint. Gap 5 (3 weeks) for the interpretability paper.

---

## New Papers to Add to `docs/references.bib`

```bibtex
@misc{ruler2024,
  title={RULER: What's the Real Context Size of Your Long-Context Language Models?},
  author={Hsieh, Cheng-Ping and Sun, Simeng and Kriman, Samuel and Acharya, Shantanu and Rekesh, Dima and Jia, Fei and Zhang, Yang and Ginsburg, Boris},
  year={2024},
  eprint={2404.06654},
  archivePrefix={arXiv}
}

@misc{babilong2024,
  title={BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack},
  author={Kuratov, Yuri and Bulatov, Aydar and Anokhin, Petr and Rodkin, Ivan and Sorokin, Dmitry and Sorokin, Artyom and Burtsev, Mikhail},
  year={2024},
  eprint={2406.10149},
  archivePrefix={arXiv}
}

@misc{longbenchv22024,
  title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks},
  author={Bai, Yushi and Tu, Shangqing and Zhang, Jiajie and Peng, Hao and Wang, Xiaozhi and Lv, Xin and Cao, Shulin and Xu, Jiazheng and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  year={2024},
  eprint={2412.15204},
  archivePrefix={arXiv}
}

@misc{sparsesinkhorn2020,
  title={Sparse Sinkhorn Attention},
  author={Tay, Yi and Bahri, Dara and Yang, Liu and Metzler, Donald and Juan, Da-Cheng},
  year={2020},
  eprint={2002.11296},
  archivePrefix={arXiv}
}

@misc{gumbelsinkhorn2018,
  title={Learning Latent Permutations with Gumbel-Sinkhorn Networks},
  author={Mena, Gonzalo and Belanger, David and Linderman, Scott and Snoek, Jasper},
  year={2018},
  eprint={1802.08665},
  archivePrefix={arXiv}
}

@misc{espformer2025,
  title={ESPFormer: Doubly-Stochastic Attention with Expected Sliced Transport Plans},
  author={Shahbazi, Ashkan and Akbari, Elaheh and Salehi, Darian and Liu, Xinran and Naderializadeh, Navid and Kolouri, Soheil},
  year={2025},
  eprint={2502.07962},
  archivePrefix={arXiv}
}

@misc{lotformer2025,
  title={LOTFormer: Doubly-Stochastic Linear Attention via Low-Rank Optimal Transport},
  author={Shahbazi, Ashkan and Thrash, Chayne and Bai, Yikun and Hamm, Keaton and NaderiAlizadeh, Navid and Kolouri, Soheil},
  year={2025},
  eprint={2509.23436},
  archivePrefix={arXiv}
}

@misc{tropicalgeometry2026,
  title={Expressivity of Transformers: A Tropical Geometry Perspective},
  author={Su, Ye and Liu, Yong},
  year={2026},
  eprint={2604.14727},
  archivePrefix={arXiv}
}

@misc{hola2026,
  title={HOLA: Hippocampal Linear Attention},
  author={Cui, Wanyun},
  year={2026},
  eprint={2607.02303},
  archivePrefix={arXiv}
}

@misc{carve2026,
  title={CARVE: Content-Aware Recurrent with Value Efficiency},
  author={Dutta, Sayak},
  year={2026},
  eprint={2606.27229},
  archivePrefix={arXiv}
}

@misc{eda2026,
  title={Erase-then-Delta Attention},
  author={Li, Xiao and Zhang, Chengruidong and Liu, Dayiheng and Zhou, Jingren and others},
  year={2026},
  eprint={2606.26560},
  archivePrefix={arXiv}
}

@misc{hypic2026,
  title={HYPIC: Hybrid-Attention LLM Serving with Position-Independent Caching},
  author={Liu, Yifei and Liu, Yang and Li, Minghao and others},
  year={2026},
  eprint={2607.01299},
  archivePrefix={arXiv}
}

@misc{emergentcapabilities2026,
  title={Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns},
  author={Baherwani, Vatsal and Chen, Zixi and Qiu, Shikai and Wilson, Andrew Gordon and Izmailov, Pavel},
  year={2026},
  eprint={2606.25010},
  archivePrefix={arXiv}
}

@misc{spotattention2026,
  title={SpotAttention: Plug-In Block-Sparse Routing},
  author={Ahmad, Huzama and Yun, Se-Young},
  year={2026},
  eprint={2606.22874},
  archivePrefix={arXiv}
}

@misc{chessformer2026,
  title={Chessformer: A Unified Architecture for Chess Modeling},
  author={Monroe, Daniel and Chalmers, Philip and Anderson, Ashton and others},
  year={2026},
  eprint={2605.19091},
  archivePrefix={arXiv}
}

@misc{rankdecay2026,
  title={Sinkhorn Doubly Stochastic Attention Rank Decay Analysis},
  author={Lapenna, Michela and Fioresi, Rita and Gharesifard, Bahman},
  year={2026},
  eprint={2604.07925},
  archivePrefix={arXiv}
}

@misc{mecw2025,
  title={Context Is What You Need: The Maximum Effective Context Window for Real World Limits of LLMs},
  author={Paulsen, Norman},
  year={2025},
  eprint={2509.21361},
  archivePrefix={arXiv}
}

@misc{nllguided2026,
  title={NLL-Guided Full-Attention Layer Selection},
  author={Tang, Qiong and Hu, Xiangkun and Liu, Xiangyang and Chen, Yiran and Shao, Yunfan},
  year={2026},
  eprint={2606.27791},
  archivePrefix={arXiv}
}

@misc{flashmorph2026,
  title={FlashMorph: Transformer-to-Hybrid Conversion},
  author={Lan, Disen and Zheng, Jianbin and Qiu, Xipeng and Cheng, Yu and others},
  year={2026},
  eprint={2606.30562},
  archivePrefix={arXiv}
}

@misc{string2024,
  title={Why Does the Effective Context Length of LLMs Fall Short?},
  author={An, Chenxin and Zhang, Jun and Zhong, Ming and Li, Lei and Gong, Shansan and Luo, Yao and Xu, Jingjing and Kong, Lingpeng},
  year={2024},
  eprint={2410.18745},
  archivePrefix={arXiv}
}
```

---

*This plan is based on a systematic review of 34 papers from arXiv (2020-2026) across 4 research areas directly relevant to Bubble Transformer's architecture, evaluation, and positioning.*
