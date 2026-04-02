# experiments/ — Bubble Transformer Research Pipeline
#
# Modules are imported via sibling imports (sys.path manipulation in run_experiment.py).
# No package-level imports needed — each module is independently runnable.
#
# Core pipeline:
#   config.py → plateau_attention.py → metrics.py → epsilon_sweep.py → visualize.py
#
# Embedding extraction:
#   extract_embeddings.py (real Qwen) / generate_mock_embeddings.py (synthetic)
#
# Utilities:
#   tensor_compat.py (NumPy fallback for PyTorch API)
#   analyze_results.py (post-hoc analysis)
#   optimal_config.py (auto-generated from sweep)
