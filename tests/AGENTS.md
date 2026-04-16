# tests/ — Unit Tests for Bubble Transformer

**7 test files** — unittest framework, PyTorch tensors for test fixtures.

---

## STRUCTURE

```
tests/
├── test_attention.py      # PlateauAttentionMechanism + Block
├── test_metrics.py        # 6 concentration/geometry metrics
├── test_cost_functions.py # Cost function variants (L2, cosine, hybrid)
├── test_dual_head.py      # Dual-head tension experiments
├── test_layer_selection.py # Layer scoring + selection logic
├── test_integration.py    # End-to-end pipeline tests
├── test_real_embeddings.py # Qwen model extraction (requires GPU)
└── __init__.py
```

---

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Attention unit tests | `test_attention.py` | `TestPlateauAttentionMechanism` — shape, sparsity, entropy |
| Metrics unit tests | `test_metrics.py` | `TestEffectiveRank`, `TestIntrinsicDimensionMLE`, etc. |
| Cost functions | `test_cost_functions.py` | CostFactory, distance metrics |
| Integration tests | `test_integration.py` | Full pipeline mock mode |
| GPU-required tests | `test_real_embeddings.py` | Skipped if no CUDA |

---

## CONVENTIONS

- **Framework**: `unittest` (not pytest) — matches CI config
- **Import pattern**: `sys.path.insert(0, "../experiments")` — sibling imports
- **Fixtures**: PyTorch tensors in `setUp()` — NumPy only in production code
- **Assertions**: `self.assertEqual`, `self.assertGreater`, `self.assertTrue`
- **Test isolation**: Each test creates fresh instances in `setUp()`
- **GPU tests**: Skip with `@unittest.skipIf(not torch.cuda.is_available(), ...)`

---

## ANTI-PATTERNS

- **DO NOT** use pytest fixtures or markers — unittest only
- **DO NOT** import NumPy in tests (use PyTorch tensors)
- **DO NOT** modify `sys.path` after imports — insert at top
- **DO NOT** skip tests without `@unittest.skipIf` decorator
- **DO NOT** use hardcoded seeds in tests — fixed in production code

---

## COMMANDS

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run single test file
python -m unittest tests.test_attention -v
python -m unittest tests.test_metrics -v

# Run with pytest (alternative)
python -m pytest tests/ -v

# Run specific test class
python -m unittest tests.test_attention.TestPlateauAttentionMechanism -v

# Run specific test method
python -m unittest tests.test_attention.TestPlateauAttentionMechanism.test_output_shape -v
```

---

## NOTES

- **CI runs**: Python 3.12 + 3.13 matrix (`.github/workflows/ci.yml`)
- **GPU tests**: Automatically skipped in CI (no CUDA)
- **Mock mode**: `test_integration.py` runs without GPU or model
- **Real embeddings**: `test_real_embeddings.py` requires Qwen3-0.6B + GPU
