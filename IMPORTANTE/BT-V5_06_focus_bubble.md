# BT V5 · Paper VI — Focus Bubble Architecture
### Sinkhorn Token Grouping + Softmax Within Groups

> **Qué es esto.** La arquitectura Focus Bubble V5 que pasa el gate de PPL ≤ 2% por primera vez en BT V5. Documenta el insight de Focus (arXiv:2604.03260) aplicado a BT: usar Sinkhorn para agrupar tokens (no para normalizar attention), luego aplicar softmax estándar dentro de cada grupo para preservar peakedness.
>
> **Tags:** `[ARCH]` arquitectura · `[BENCH]` benchmark · `[INNOVATION]` contribución novel · `[INVARIANT]` preservado · `⚠` · `☐` acción.

---

## §1 — El cambio de paradigma

### §1.1 El problema con SIRI (V1-V4)

Las versiones anteriores de BT V5 usaban SIRI para **reemplazar** softmax:

```
SIRI:  softmax(QK^T) → exp(-C/ε) → Sinkhorn → doubly-stochastic
```

El resultado: doubly-stochastic attention **destruye peakedness**. Para LM se necesita distribución sharp sobre tokens relevantes. Forzar doubly-stochastic distribuye masa uniformemente → PPL +30% a +211%.

### §1.2 El insight de Focus

El paper Focus (arXiv:2604.03260) demuestra que Sinkhorn puede **mejorar** PPL si se usa para **agrupar** tokens, no para normalizar attention:

```
Focus:  softmax(QK^T) → Sinkhorn(agrupar) → softmax(dentro de grupos)
```

Resultado: peakedness preservado + estructura geométrica agregada. Focus logra 42.8 → 30.3 PPL (29% mejora) en GPT-2 124M.

### §1.3 Aplicación a BT V5

BT V5 tenía SIRI y Power Diagram ψ como componentes centrales. La pregunta: ¿podemos preservar estos componentes y aún así pasar el gate?

**Respuesta: SÍ**, restringiendo el rol de Sinkhorn a grouping:

```
Focus Bubble:
  1. Q, K, V projections
  2. S = Q @ K^T / sqrt(d)  (dot-product, NO geométrico)
  3. S = S + ψ  (Power Diagram bias)
  4. groups = Sinkhorn(S, tau)  (SIRI usado para agrupar)
  5. attn = softmax(S + log(groups))  (softmax estándar, peakedness preservado)
  6. output = attn @ V
```

**Invariantes preservados**:
- ✅ SIRI: usado para token grouping (Sinkhorn iterations)
- ✅ Power Diagram ψ: bias en scores (absorbido por normalización columnar)
- ✅ Softmax peakedness: mantenido vía softmax estándar
- ✅ DeltaNet: compatible vía interpolación (FocusDeltaNet variant)

**Invariantes violados**:
- ⚠ I1 (geometric cost): Focus usa dot-product, no ||Q-K||². Esto es intencional y necesario.
- ⚠ I2 (doubly-stochastic attention): Sinkhorn produce doubly-stochastic, pero se usa como grouping mask, no como attention weights.

---

## §2 — Arquitectura detallada

### §2.1 FocusBubbleAttention (core)

```python
class FocusBubbleAttention(nn.Module):
    def forward(self, x):
        B, N, D = x.shape
        
        # 1. Projections
        Q = self.q_proj(x).view(B, N, H, D_h).transpose(1, 2)  # [B, H, N, D_h]
        K = self.k_proj(x).view(B, N, H, D_h).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, D_h).transpose(1, 2)
        
        # 2. Compute scores
        S = Q @ K.transpose(-2, -1) / sqrt(D_h)  # [B, H, N, N]
        S = S + self.psi  # Power Diagram bias
        
        # 3. Apply causal mask
        S = S + causal_mask
        
        # 4. Sinkhorn for token grouping (tau iterations)
        S_scaled = S * self.epsilon
        S_scaled = S_scaled.clamp(-50, 50)
        for _ in range(self.tau_iters):
            S_scaled = S_scaled - logsumexp(S_scaled, dim=-1, keepdim=True)  # row norm
            S_scaled = S_scaled - logsumexp(S_scaled, dim=-2, keepdim=True)  # col norm
        groups = exp(S_scaled)  # doubly-stochastic
        
        # 5. Softmax within groups
        attn = softmax(S + log(groups + 1e-10), dim=-1)  # peakedness preserved
        
        # 6. Output
        out = attn @ V
        out = self.o_proj(out)
        return out
```

### §2.2 FocusBubbleDeltaNet (best variant)

Combina Focus grouping con DeltaNet delta rule:

```python
class FocusBubbleDeltaNet(nn.Module):
    def forward(self, x):
        # ... projections ...
        
        # DeltaNet base (O(N) linear)
        Q_safe = safe_normalize(Q)  # prevent overflow
        K_safe = safe_normalize(K)
        V_safe = safe_normalize(V)
        out_delta = deltanet_forward(Q_safe, K_safe, V_safe)
        
        # Focus grouping
        out_focus = focus_bubble_attention(x)
        
        # Interpolation
        out = self.lam * out_delta + (1 - self.lam) * out_focus
        return out
```

**Safe normalization** (crítico para evitar NaN):

```python
def _safe_normalize(x, eps=1e-6):
    norm = x.norm(dim=-1, keepdim=True)
    return x / torch.clamp(norm, min=eps)
```

---

## §3 — Resultados experimentales

### §3.1 Gate PASSED

| Configuración | PPL | Δ% | Gate |
|---------------|-----|-----|------|
| Baseline (softmax) | 22.513 | — | — |
| **FocusDeltaNet L7 (λ=0.3)** | **22.550** | **+0.16%** | **PASS** |
| Focus L7 (λ=0.0) | 22.681 | +0.74% | PASS |
| Focus L12 (λ=0.0) | 22.706 | +0.86% | PASS |
| Previous best (Hybrid 3) | 23.052 | +2.39% | FAIL |

**Primera configuración BT V5 en pasar el gate.**

### §3.2 Layer sweep (eps=0.001, tau=1)

| Layer | PPL | Δ% | Gate |
|-------|-----|-----|------|
| L3 | 22.960 | +1.98% | PASS |
| L5 | 22.928 | +1.84% | PASS |
| **L7** | **22.681** | **+0.74%** | **PASS** |
| L9 | 23.026 | +2.28% | **FAIL** |
| L10 | 22.757 | +1.08% | PASS |
| L11 | 22.772 | +1.15% | PASS |
| L12 | 22.706 | +0.86% | PASS |
| L15 | 22.814 | +1.33% | PASS |
| L19 | 22.850 | +1.50% | PASS |
| L23 | 23.154 | +2.84% | **FAIL** |

**8 de 10 capas pasan el gate.** L7 es óptimo.

### §3.3 Lambda sweep (L7, FocusDeltaNet)

| Lambda | PPL | Δ% | Gate |
|--------|-----|-----|------|
| 0.0 (Focus only) | 22.681 | +0.74% | PASS |
| 0.2 | 22.554 | +0.18% | PASS |
| **0.3** | **22.550** | **+0.16%** | **PASS** |
| 0.5 | 22.651 | +0.61% | PASS |
| 0.7 | 22.900 | +1.72% | PASS |
| 0.8 | 23.079 | +2.51% | FAIL |
| 1.0 (DeltaNet only) | 23.558 | +4.64% | FAIL |

**Lambda=0.3 es óptimo** (30% DeltaNet, 70% Focus).

---

## §4 — L9 anomaly

L9 es la única capa media que falla el gate (+2.28%) mientras vecinas L10/L11/L12 pasan (+1.08%, +1.15%, +0.86%).

### §4.1 Re-measurement

L9 re-medido en aislamiento: PPL=23.026, Δ=+2.28%. Diferencia del original: 0.0002 PPL points. **Replica exacto** — no es ruido.

### §4.2 Análisis geométrico

L9 tiene el CR_focus más bajo (0.163) y CR_diff más negativo (-0.063). Sin embargo, L2_ratio Focus-vs-Softmax (24.00%) no es outlier — L10 (32.22%) y L11 (37.28%) tienen ratios más altos y pasan el gate.

### §4.3 Per-head analysis

Outlier heads en L9 (L2_ratio > mean + 1.5*std):
- Head 2: L2_ratio=131.81%
- Head 6: L2_ratio=153.63%

L10 tiene outliers diferentes (Head 6, Head 9), sugiriendo que el fallo de L9 no es un issue genérico de head-geometry.

**Conclusión**: El fallo de L9 parece venir de efectos de integración entre heads, no de geometría individual.

---

## §5 — Power Diagram ψ: honest caveat

### §5.1 El claim

Docs anteriores de BT V5 claimaban que Power Diagram ψ proporciona inyección de estructura geométrica.

### §5.2 La realidad

El bias ψ es **absorbido por la normalización columnar de Sinkhorn**. Empíricamente:

- L12 FocusOnly psi=True: PPL=23.082
- L12 FocusOnly psi=False: PPL=23.082
- **Resultados idénticos** — ψ no tiene efecto

### §5.3 Qué significa esto

Power Diagram ψ no proporciona valor independiente en Focus Bubble. Se preserva en el código por compatibilidad y futura investigación, pero no contribuye al resultado gate-passing.

Esto es consistente con el honest caveat en WHITEPAPER v2.0 §7.4.

---

## §6 — Limitaciones y trabajo futuro

### §6.1 Limitaciones

1. **Single-layer swap only**: No validado con pretrain from scratch
2. **Short context only**: Testeado en N=256 tokens, no long-context
3. **No downstream evaluation**: MMLU, HumanEval, GSM8K no probados
4. **No speed comparison**: Wall-clock time vs softmax no medido
5. **L9 anomaly sin explicar**: Root cause no identificado

### §6.2 Trabajo futuro

1. **Pretrain Bubble-1.3B** from scratch con Focus Bubble
2. **Long-context evaluation** (NIAH, RULER) en 4K-32K tokens
3. **Downstream task evaluation** (MMLU, HumanEval, GSM8K)
4. **Speed optimization** (CUDA kernel para Focus Bubble)
5. **L9 root cause investigation** (per-head attention pattern visualization)
6. **Multi-layer Focus Bubble** (reemplazar múltiples capas simultáneamente)

### §6.3 Preguntas abiertas

1. ¿Focus Bubble sobrevive pretrain from scratch?
2. ¿Focus Bubble mantiene retrieval accuracy en 32K context?
3. ¿Focus Bubble es más rápido o más lento que softmax en wall-clock time?
4. ¿Focus Bubble puede combinarse con otras variantes de linear attention (Gated DeltaNet, KDA)?

---

## §7 — Reproducibilidad

### §7.1 Comandos

```bash
# Fine epsilon sweep
py experiments/benchmark_focus_fine_sweep.py

# Layer sweep at optimal params
py experiments/benchmark_focus_layer_sweep_optimal.py

# FocusDeltaNet lambda sweep
py experiments/benchmark_focus_deltanet_sweep.py
```

### §7.2 Resultados esperados

| Benchmark | Expected best | JSON file |
|-----------|---------------|-----------|
| Fine sweep | L12 eps=0.001 tau=1: PPL=22.706 | `results_real/focus_bubble/focus_fine_sweep.json` |
| Layer sweep | L7 eps=0.001 tau=1: PPL=22.681 | `results_real/focus_bubble/focus_layer_sweep_optimal.json` |
| FocusDeltaNet | L7 lambda=0.3: PPL=22.550 | `results_real/focus_bubble/focus_deltanet_sweep.json` |

### §7.3 Hardware

- GPU: GTX 1650 (4.3GB VRAM) o mejor
- Tiempo: ~30 minutos para suite completa

---

## §8 — Referencias

### Papers externos

- Focus (arXiv:2604.03260) - Sinkhorn for token grouping, softmax within groups
- Gated DeltaNet (arXiv:2412.06464) - Delta rule + gating
- Kimi Linear (arXiv:2510.26692) - Hybrid linear attention
- Sinkformer (Sander 2022) - Doubly-stochastic attention

### Docs internos BT V5

- `BT-V5_00_bases_primarias.md` - Axiomas y teoremas
- `BT-V5_01_architecture.md` - Spec de arquitectura V5 (SIRI-based)
- `BT-V5_02_siri_theory.md` - Teoría SIRI
- `BT-V5_05_protocol_positioning.md` - Gate de PPL

### Código

- `experiments/focus_bubble_attention.py` - Core FocusBubbleAttention + FocusBubbleDeltaNet
- `experiments/qwen3_focus_bubble_wrapper.py` - Qwen3 drop-in wrapper
- `tests/test_focus_bubble_*.py` - Test suite

---

*BT-V5_06_focus_bubble.md v1.0 - Julio 2026*
*Focus Bubble V5: GATE PASSED (+0.16% PPL at L7 lambda=0.3)*
*Status: Research artifact, not production system*
*Next milestone: Pretrain from scratch validation*
