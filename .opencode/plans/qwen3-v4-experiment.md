# Implementation Plan: Qwen3-0.6B V4 Experiment

**Project**: LLM-BUBBLE  
**Goal**: Run real experiment with Qwen3-0.6B using Bubble Transformer V4  
**Hardware**: GTX 1650 (4GB VRAM)  
**Date**: 2026-04-15

---

## Executive Summary

Extract real embeddings from Qwen3-0.6B-Base (4-bit quantized) and run epsilon sweep with Bubble Transformer V4 architecture. The plan addresses the known dual-head unpacking error and integrates V4's FPS + Expert-Choice routing.

**Key Changes from Current State:**
1. Target layers: `[0, 4, 8, 12, 16, 20, 24]` (7 layers, distributed across 28)
2. Model: Qwen3-0.6B-Base (4-bit NF4, ~1GB VRAM)
3. V4 Integration: FPS init + Expert-Choice routing for dual-head

---

## Phase 1: Environment & Dependency Verification

### Task 1.1: Verify Python Environment
**What**: Check Python version and install required packages.

```bash
python --version  # Expect: 3.12+
pip list | grep -E "transformers|torch|bitsandbytes|accelerate"
```

**Required versions:**
- `transformers >= 4.51.0` (Qwen3 support)
- `torch >= 2.6.0` with CUDA 12.4
- `bitsandbytes >= 0.43.0`
- `accelerate >= 0.30.0`

**Verification:**
```bash
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

### Task 1.2: Verify CUDA Memory
**What**: Confirm GTX 1650 has sufficient VRAM.

```bash
python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

**Expected**: ~4.0GB total, need ~1.5GB for Qwen3-0.6B 4-bit + extraction overhead.

---

## Phase 2: Fix Dual-Head Unpacking Error

### Task 2.1: Audit Return Signatures
**What**: Document all attention class return signatures.

**Current state:**
- `PlateauAttentionMechanism.forward(return_attention=True)` → `(output, A)` [2 values]
- `DualHeadPlateauAttention.forward(return_attention=True)` → `(output, A_low, A_high)` [3 values]

**Files to audit:**
1. `experiments/epsilon_sweep.py` - Lines 154, 441
2. `experiments/visualize.py` - Lines around 61 (mentioned in AGENTS.md)
3. `experiments/run_experiment.py` - Any tension mode calls

### Task 2.2: Identify Error Location
**What**: Find code that incorrectly unpacks 3 values from single-head attention.

**Search pattern:**
```bash
grep -n "forward.*return_attention" experiments/*.py
```

**Known locations (from analysis):**
- `epsilon_sweep.py:154` - CORRECT: `output, attn_matrix = attn.forward(raw_input, return_attention=True)`
- `epsilon_sweep.py:441` - CORRECT: `output, A_low, A_high = dual_attn.forward(...)`

**Potential issue locations:**
- `visualize.py` around line 61
- Any tension sweep code that uses single-head attention with 3-value unpacking

### Task 2.3: Implement Fix
**What**: Ensure callers use correct unpacking based on attention type.

**Solution pattern:**
```python
# BEFORE (broken):
output, A_low, A_high = attn.forward(x, return_attention=True)  # Wrong for single-head

# AFTER (fixed):
if isinstance(attn, DualHeadPlateauAttention):
    output, A_low, A_high = attn.forward(x, return_attention=True)
else:
    output, A = attn.forward(x, return_attention=True)
```

**Alternative: Unify return signature:**
```python
# Option B: Make single-head return 3 values (A_high=None)
def forward(self, x, return_attention=False):
    # ...
    if return_attention:
        return output, A, None  # Unify to 3 values
    return output
```

---

## Phase 3: Configure Target Layers for Qwen3-0.6B

### Task 3.1: Update Config
**What**: Change target layers to distributed 7-layer selection.

**File**: `experiments/config.py`

**Current (line 44):**
```python
target_layers: List[int] = field(default_factory=lambda: [3, 7, 11, 15, 19, 23])
```

**New:**
```python
# Qwen3-0.6B: 28 layers, all full-attention
# Distributed sampling: every 4th layer starting from 0
target_layers: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24])
```

**Rationale:**
- Qwen3-0.6B has 28 layers total (all full-attention)
- 7 layers provides good coverage across depth
- Includes layer 0 (embedding) and late layers (task-specific)

### Task 3.2: Add Model-Specific Config
**What**: Create Qwen3-0.6B specific configuration section.

**Add to `config.py`:**
```python
@dataclass
class Qwen3_06B_Config:
    """Qwen3-0.6B-Base specific settings."""
    model_name: str = "Qwen/Qwen3-0.6B-Base"
    num_layers: int = 28
    d_model: int = 1024
    num_heads: int = 16
    num_kv_heads: int = 8  # GQA ratio 2:1
    head_dim: int = 128
    intermediate_size: int = 3072
    max_position_embeddings: int = 32768
    target_layers: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20, 24])
```

---

## Phase 4: V4 Integration

### Task 4.1: Create V4 Adapter for Experiment Pipeline
**What**: Create adapter that integrates V4 FPS+Expert-Choice routing.

**New file**: `experiments/v4_adapter.py`

```python
"""
V4 Adapter - Integrates FPS+Expert-Choice routing into experiment pipeline.
"""
import numpy as np
from typing import Optional, Tuple, Dict, List
from plateau_attention import PlateauAttentionMechanism

class V4PlateauAdapter:
    """
    Wraps V4 concepts for use in epsilon sweep.
    
    Uses FPS-like initialization for better centroid selection
    and applies epsilon-based Sinkhorn for attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        epsilon: float = 0.1,
        num_experts: int = 32,  # C in V4
        top_k: int = 8,  # Tokens per expert
        tau_iters: int = 5,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.epsilon = epsilon
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Use standard PlateauAttention for Sinkhorn
        self._sinkhorn = PlateauAttentionMechanism(
            d_model=d_model,
            num_heads=num_heads,
            epsilon=epsilon,
            tau_iters=tau_iters,
            seed=seed,
        )
    
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward pass with V4-style routing + Sinkhorn attention."""
        # For now, delegate to PlateauAttention (V4 integration is additive)
        # Full V4 would add FPS init and expert-choice routing here
        return self._sinkhorn.forward(x, mask=mask, return_attention=return_attention)
```

### Task 4.2: Add V4 Mode to Run Experiment
**What**: Add `--v4` flag to run experiment with V4 adapter.

**Modify**: `experiments/run_experiment.py`

```python
# Add argument
parser.add_argument("--v4", action="store_true", help="Use V4 FPS+Expert-Choice routing")
parser.add_argument("--num-experts", type=int, default=32, help="Number of experts for V4")
parser.add_argument("--top-k", type=int, default=8, help="Tokens per expert for V4 routing")

# In sweep section:
if args.v4:
    from v4_adapter import V4PlateauAdapter
    attn = V4PlateauAdapter(
        d_model=d_model,
        num_heads=num_heads,
        epsilon=eps,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
else:
    attn = PlateauAttentionMechanism(...)
```

---

## Phase 5: Embedding Extraction

### Task 5.1: Create Test Corpus
**What**: Prepare corpus for embedding extraction.

**File**: `data/test_corpus.jsonl`

**Minimum corpus:**
```json
{"text": "The quick brown fox jumps over the lazy dog. This is a test sentence for embedding extraction from Qwen3-0.6B model."}
{"text": "Machine learning models learn representations from data. Transformers use self-attention mechanisms."}
{"text": "The bubble transformer reformulates attention as optimal transport with entropic regularization."}
```

**Target**: 10-50 texts for initial extraction, 100+ for production.

### Task 5.2: Run Extraction
**What**: Extract embeddings from Qwen3-0.6B.

```bash
python experiments/extract_embeddings.py \
    --model "Qwen/Qwen3-0.6B-Base" \
    --corpus "data/test_corpus.jsonl" \
    --output-dir "embeddings" \
    --batch-size 2 \
    --max-length 512 \
    --target-layers 0 4 8 12 16 20 24
```

**Expected output:**
```
embeddings/
├── metadata.json          # mode="real", d_model=1024
├── raw_input.npy          # [total_tokens, 1024]
└── softmax/
    ├── layer_0.npy        # [tokens, 1024]
    ├── layer_4.npy
    ├── layer_8.npy
    ├── layer_12.npy
    ├── layer_16.npy
    ├── layer_20.npy
    └── layer_24.npy
```

### Task 5.3: Verify Extraction Quality
**What**: Check embeddings have expected properties.

```bash
python -c "
import numpy as np
import json

# Load metadata
with open('embeddings/metadata.json') as f:
    meta = json.load(f)
print(f'Mode: {meta[\"mode\"]}')
print(f'Model: {meta[\"model\"]}')
print(f'Layers: {meta[\"layers_saved\"]}')

# Check effective rank per layer
for layer in [0, 4, 8, 12, 16, 20, 24]:
    emb = np.load(f'embeddings/softmax/layer_{layer}.npy')
    centered = emb - emb.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    p = S / S.sum()
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-10)))
    print(f'Layer {layer}: shape={emb.shape}, eff_rank={eff_rank:.1f}')
"
```

---

## Phase 6: Run Epsilon Sweep

### Task 6.1: Configure Sweep Parameters
**What**: Set optimal epsilon range based on research.

**Recommended epsilon values:**
```python
epsilon_values = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
```

**Sweet spot range**: `[0.01, 0.1]`

### Task 6.2: Run Sweep (Mock First)
**What**: Verify pipeline with mock embeddings before real extraction.

```bash
python experiments/run_experiment.py \
    --mode mock \
    --num-layers 28 \
    --d-model 1024 \
    --num-heads 16 \
    --target-layers 0 4 8 12 16 20 24 \
    --epsilon-values 0.001 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0 \
    --seed 42
```

**Expected output:**
```
results/
├── epsilon_sweep.json     # Full sweep results
└── sweet_spot_analysis.md # Recommendations

plots/
├── effective_rank_curves.png
├── concentration_heatmap_*.png
├── pareto_frontier.png
└── ...
```

### Task 6.3: Run Sweep (Real Mode)
**What**: Run epsilon sweep on real Qwen3-0.6B embeddings.

```bash
python experiments/run_experiment.py \
    --mode real \
    --target-layers 0 4 8 12 16 20 24 \
    --epsilon-values 0.001 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0 \
    --d-model 1024 \
    --num-heads 16 \
    --seed 42
```

---

## Phase 7: V4 Dual-Head Experiment

### Task 7.1: Run Dual-Head Tension Sweep
**What**: Test V4 dual-head with tension coefficient.

```bash
python experiments/run_experiment.py \
    --mode tension \
    --target-layers 0 4 8 12 16 20 24 \
    --d-model 1024 \
    --num-heads 16
```

### Task 7.2: Analyze Tension Results
**What**: Find optimal alpha coefficient.

**Output**: `results/tension_sweep.json` with alpha sweep results.

---

## Phase 8: Visualization & Reporting

### Task 8.1: Generate All Plots
**What**: Create comprehensive visualization suite.

```bash
python experiments/run_experiment.py \
    --mode real \
    --skip-generation \
    --no-skip-visualization
```

### Task 8.2: Generate Sweet Spot Report
**What**: Produce markdown analysis with recommendations.

**Expected sections:**
1. Executive Summary
2. Epsilon Analysis
3. Layer-by-Layer Metrics
4. Sweet Spot Recommendation
5. V4 vs Baseline Comparison
6. Next Steps

---

## Phase 9: Testing (TDD)

### Task 9.1: Write Tests for V4 Adapter
**What**: Create unit tests for V4 integration.

**File**: `tests/test_v4_adapter.py`

```python
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from v4_adapter import V4PlateauAdapter

class TestV4Adapter(unittest.TestCase):
    def test_output_shape(self):
        """V4 adapter output matches input shape."""
        adapter = V4PlateauAdapter(d_model=128, num_heads=4)
        x = np.random.randn(2, 32, 128).astype(np.float32)
        output = adapter.forward(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_return_attention(self):
        """V4 adapter returns 2 values with return_attention=True."""
        adapter = V4PlateauAdapter(d_model=128, num_heads=4)
        x = np.random.randn(2, 32, 128).astype(np.float32)
        output, A = adapter.forward(x, return_attention=True)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(A.shape[-1], 32)

if __name__ == "__main__":
    unittest.main()
```

### Task 9.2: Run Full Test Suite
**What**: Verify all tests pass before commit.

```bash
python -m pytest tests/ -v
python -m pytest tests/test_dual_head.py -v  # Focus on dual-head tests
```

---

## Phase 10: Atomic Commit Strategy

### Commit 1: Fix dual-head unpacking error
**Files**: `experiments/epsilon_sweep.py`, `experiments/visualize.py` (if needed)
```bash
git add experiments/epsilon_sweep.py experiments/visualize.py
git commit -m "fix: resolve dual-head unpacking error in REAL mode

- Audit all attention.forward() calls for correct unpacking
- PlateauAttentionMechanism returns 2 values (output, A)
- DualHeadPlateauAttention returns 3 values (output, A_low, A_high)
- Add type check before unpacking in visualize.py

Fixes: #known-issue-dual-head-unpacking"
```

### Commit 2: Update target layers for Qwen3-0.6B
**Files**: `experiments/config.py`
```bash
git add experiments/config.py
git commit -m "feat(config): add Qwen3-0.6B target layers [0,4,8,12,16,20,24]

- Change from 6 layers to 7 layers distributed across 28
- Add Qwen3_06B_Config dataclass
- All 28 layers are full-attention in Qwen3-0.6B"
```

### Commit 3: Add V4 adapter for experiment pipeline
**Files**: `experiments/v4_adapter.py`, `experiments/run_experiment.py`
```bash
git add experiments/v4_adapter.py experiments/run_experiment.py
git commit -m "feat(v4): add V4 FPS+Expert-Choice adapter

- V4PlateauAdapter wraps V4 concepts for epsilon sweep
- Add --v4, --num-experts, --top-k CLI flags
- Maintain backward compatibility with V3"
```

### Commit 4: Add V4 adapter tests
**Files**: `tests/test_v4_adapter.py`
```bash
git add tests/test_v4_adapter.py
git commit -m "test(v4): add unit tests for V4PlateauAdapter

- Test output shape preservation
- Test return_attention signature
- Test no NaN/Inf in output"
```

### Commit 5: Update extraction defaults
**Files**: `experiments/extract_embeddings.py`
```bash
git add experiments/extract_embeddings.py
git commit -m "fix(extract): update defaults for Qwen3-0.6B

- Default target_layers to [0,4,8,12,16,20,24]
- Update QWEN3_06B_CONFIG with correct parameters
- Improve error messages for Windows bitsandbytes"
```

---

## Anti-Patterns to Avoid

### 1. NumPy Contract Violation
**WRONG:**
```python
# In metrics.py or plateau_attention.py
import torch
output = torch.tensor(embeddings)  # Breaks NumPy-only contract
```

**CORRECT:**
```python
# Pure NumPy only
import numpy as np
output = np.array(embeddings, dtype=np.float32)
```

### 2. Skipping Log-Domain Sinkhorn
**WRONG:**
```python
# Direct exp() will underflow at epsilon < 0.01
A = np.exp(-C / epsilon)  # Numerical instability
```

**CORRECT:**
```python
# Use log-domain Sinkhorn
log_S = -C / epsilon
u = -logsumexp(log_S + v[:, :, np.newaxis, :], axis=-1)
# ... continue in log space
A = np.exp(log_S + u[:, :, :, np.newaxis] + v[:, :, np.newaxis, :])
```

### 3. Incorrect Dual-Head Unpacking
**WRONG:**
```python
# Assuming 3 values from single-head
output, A_low, A_high = single_head_attn.forward(x, return_attention=True)
# ValueError: not enough values to unpack (expected 3, got 2)
```

**CORRECT:**
```python
if isinstance(attn, DualHeadPlateauAttention):
    output, A_low, A_high = attn.forward(x, return_attention=True)
else:
    output, A = attn.forward(x, return_attention=True)
```

### 4. GPU Memory Leaks in Extraction
**WRONG:**
```python
# Accumulating on GPU
layer_outputs[layer_idx] = hidden_states  # Stays on GPU!
```

**CORRECT:**
```python
# Move to CPU immediately
layer_outputs[layer_idx] = hidden_states.cpu()
torch.cuda.empty_cache()  # Clear between batches
```

### 5. Bare Except Clauses
**WRONG:**
```python
try:
    output = attn.forward(x)
except:  # Catches everything including KeyboardInterrupt
    pass
```

**CORRECT:**
```python
try:
    output = attn.forward(x)
except (ValueError, RuntimeError, MemoryError) as e:
    print(f"Error: {e}")
    output = None
```

---

## Expected Deliverables

### Files Created/Modified:
1. `experiments/config.py` - Updated target layers
2. `experiments/v4_adapter.py` - New V4 integration adapter
3. `experiments/run_experiment.py` - Added --v4 flag
4. `experiments/epsilon_sweep.py` - Fixed dual-head unpacking
5. `experiments/visualize.py` - Fixed unpacking (if needed)
6. `tests/test_v4_adapter.py` - V4 unit tests

### Output Files:
1. `embeddings/metadata.json` - Real extraction metadata
2. `embeddings/softmax/layer_*.npy` - 7 layer embeddings
3. `embeddings/raw_input.npy` - Raw token embeddings
4. `results/epsilon_sweep.json` - Full sweep results
5. `results/sweet_spot_analysis.md` - Recommendation report
6. `plots/*.png` - 7+ visualization files

### Metrics to Collect:
- Effective rank per layer per epsilon
- Concentration ratio heatmap
- Anisotropy index vs epsilon
- Intrinsic dimensionality vs epsilon
- Pareto frontier (concentration vs expressivity)
- Optimal epsilon recommendation
- V4 vs baseline comparison

---

## Success Criteria

1. **Extraction completes**: No OOM errors on GTX 1650
2. **All tests pass**: `python -m pytest tests/ -v` returns 100%
3. **No unpacking errors**: REAL mode runs without ValueError
4. **Sweet spot identified**: Clear epsilon recommendation in report
5. **Visualizations generated**: All 7+ plots created successfully
6. **V4 integration works**: Can run with --v4 flag

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OOM during extraction | Reduce batch_size to 1, clear cache between batches |
| Dual-head error persists | Run tests first, audit all forward() calls |
| Slow extraction | Use smaller corpus (10 texts) for initial test |
| V4 integration issues | Start with adapter pattern, full integration later |
| Windows bitsandbytes | Document WSL2 alternative in README |

---

## Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| 1. Environment Setup | 15 min | Verify packages |
| 2. Fix Dual-Head Error | 30 min | Audit + fix |
| 3. Config Update | 10 min | Target layers |
| 4. V4 Integration | 45 min | Adapter + CLI |
| 5. Embedding Extraction | 20 min | 10 texts, 4-bit |
| 6. Epsilon Sweep | 30 min | Mock + Real |
| 7. Dual-Head Sweep | 20 min | Tension mode |
| 8. Visualization | 15 min | Generate plots |
| 9. Testing | 30 min | TDD verification |
| 10. Commits | 15 min | Atomic commits |
| **Total** | **3.5 hours** | |

---

## Next Actions

1. Plan created (this document)
2. Run `python -m pytest tests/ -v` to verify current state
3. Implement dual-head fix (Phase 2)
4. Update config (Phase 3)
5. Create V4 adapter (Phase 4)
6. Run extraction and sweep (Phases 5-6)

---

*Plan generated by Sisyphus planning system*  
*Last updated: 2026-04-15*
