"""Common mock utilities for the LLM-BUBBLE test suite.

This module provides reusable helpers for generating synthetic test data
(mock embeddings, cost matrices, attention matrices, corpus snippets)
and shared assertion utilities. All subsequent test files should import
from here to ensure consistency across the test suite.

Conventions
-----------
- PyTorch tensors are used for mock embeddings and attention matrices
  (matching existing test fixture conventions).
- NumPy arrays are used for cost matrices and corpus data (matching
  production-code conventions in experiments/).\n- Dimensions follow the project standard: B=2, N=32, D=128, num_heads=4.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import numpy as np
import torch


def create_mock_embeddings(B: int, N: int, D: int) -> torch.Tensor:
    """Generate a random PyTorch tensor of shape [B, N, D].

    Parameters
    ----------
    B : int
        Batch size.
    N : int
        Sequence length (number of tokens).
    D : int
        Embedding dimension.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``[B, N, D]`` drawn from a standard normal
        distribution.
    """
    return torch.randn(B, N, D)


def create_mock_cost_matrix(N: int, D: int) -> np.ndarray:
    """Generate a mock cost matrix of shape [N, N] using L2 squared distances.

    The matrix is computed from two random sets of points in ``R^D`` and
    reflects the same cost structure used by ``PlateauAttentionMechanism``.

    Parameters
    ----------
    N : int
        Sequence length (matrix side length).
    D : int
        Dimensionality used to generate the underlying point cloud.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``[N, N]`` containing pairwise L2 squared
        distances.
    """
    rng = np.random.RandomState(42)
    points = rng.randn(N, D).astype(np.float32)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    cost = np.sum(diff ** 2, axis=-1).astype(np.float32)
    return cost


def assert_numpy_close(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert that two NumPy arrays are element-wise close.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare.
    rtol : float, optional
        Relative tolerance (default 1e-5).
    atol : float, optional
        Absolute tolerance (default 1e-8).

    Raises
    ------
    AssertionError
        If any element differs by more than ``atol + rtol * |b|``.
    """
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(a - b))
        raise AssertionError(
            f"Arrays not close (max_diff={max_diff:.6e}, rtol={rtol}, atol={atol})"
        )


def create_mock_attention_matrix(
    B: int, num_heads: int, N: int
) -> torch.Tensor:
    """Generate a mock doubly-stochastic attention matrix.

    The returned tensor has shape ``[B, num_heads, N, N]`` and each
    ``[N, N]`` slice approximately sums to 1 across both rows and columns,
    mimicking the output of Sinkhorn-Knopp normalization.

    Parameters
    ----------
    B : int
        Batch size.
    num_heads : int
        Number of attention heads.
    N : int
        Sequence length.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``[B, num_heads, N, N]`` with values in
        ``(0, 1)`` that approximately satisfy doubly-stochastic constraints.
    """
    torch.manual_seed(42)
    # Start with uniform + small random perturbation
    attn = torch.ones(B, num_heads, N, N) + torch.randn(B, num_heads, N, N) * 0.1
    attn = torch.relu(attn) + 1e-6  # ensure positivity
    # Row-normalise
    attn = attn / attn.sum(dim=-1, keepdim=True)
    # Column-normalise (one Sinkhorn-like pass)
    attn = attn / attn.sum(dim=-2, keepdim=True)
    # Final row-normalise to guarantee row-stochastic
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


def mock_matplotlib_figure():
    """Create and return a blank Matplotlib figure for testing plotting code.

    Returns
    -------
    matplotlib.figure.Figure
        A new blank figure instance.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for headless testing
    from matplotlib import pyplot as plt

    return plt.figure()


def create_mock_corpus(num_sentences: int = 10) -> list:
    """Generate a list of sentence dicts mimicking ``data/test_corpus.jsonl``.

    Each dict has the key ``"text"`` with a synthetic sentence.  The content
    is deterministic so that tests are reproducible.

    Parameters
    ----------
    num_sentences : int, optional
        Number of sentences to generate (default 10).

    Returns
    -------
    list[dict]
        List of ``{"text": "..."}`` dictionaries.
    """
    templates = [
        "Optimal transport provides a geometrically meaningful way to compare probability distributions.",
        "The Sinkhorn algorithm scales matrices while preserving entropy constraints.",
        "Transformer attention can be viewed as a soft selection mechanism over value vectors.",
        "High-dimensional embeddings often exhibit concentration of measure phenomena.",
        "Entropic regularization smooths the optimal transport plan and enables faster computation.",
        "Doubly stochastic matrices generalise permutation matrices in a continuous relaxation.",
        "The effective rank of a matrix captures how many dimensions are actively used.",
        "Anisotropy in embedding spaces can lead to degraded retrieval performance.",
        "Plateau's laws describe the geometric structure of soap films and minimal surfaces.",
        "Viscosity coefficients control the trade-off between sparsity and smoothness in transport plans.",
        "Intrinsic dimensionality estimates reveal the true complexity of neural representations.",
        "Layer-wise analysis of transformers shows evolving geometry across depth.",
        "Bubble attention replaces softmax with Sinkhorn-Knopp for structured sparsity.",
        "The Wasserstein distance respects the metric structure of the underlying space.",
        "Cost matrices encode geometric or semantic similarity between tokens.",
        "Epsilon sweeps help identify the optimal regularisation strength for a given task.",
        "Attention entropy measures how peaked or diffuse the token weights are.",
        "Concentration ratios quantify the fraction of mass in the top-k attention entries.",
        "Neural network manifolds can be studied via pairwise distance distributions.",
        "Transport plans with low entropy favour sharp, interpretable alignments.",
    ]
    sentences = []
    for i in range(num_sentences):
        sentences.append({"text": templates[i % len(templates)]})
    return sentences


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestHelpers(unittest.TestCase):
    """Smoke tests for the utility functions in ``test_helpers.py``."""

    def setUp(self):
        """Set standard dimensions used across the test suite."""
        self.B = 2
        self.N = 32
        self.D = 128
        self.num_heads = 4

    def test_create_mock_embeddings_shape(self):
        """``create_mock_embeddings`` must return a [B, N, D] tensor."""
        emb = create_mock_embeddings(self.B, self.N, self.D)
        self.assertEqual(emb.shape, (self.B, self.N, self.D))
        self.assertEqual(emb.dtype, torch.float32)

    def test_create_mock_cost_matrix_shape(self):
        """``create_mock_cost_matrix`` must return an [N, N] array."""
        cost = create_mock_cost_matrix(self.N, self.D)
        self.assertEqual(cost.shape, (self.N, self.N))
        self.assertEqual(cost.dtype, np.float32)

    def test_create_mock_cost_matrix_non_negative(self):
        """Cost matrices should be non-negative."""
        cost = create_mock_cost_matrix(self.N, self.D)
        self.assertTrue(np.all(cost >= 0))

    def test_assert_numpy_close_passes(self):
        """``assert_numpy_close`` must succeed for identical arrays."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert_numpy_close(a, b)  # should not raise

    def test_assert_numpy_close_fails(self):
        """``assert_numpy_close`` must raise for divergent arrays."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        with self.assertRaises(AssertionError):
            assert_numpy_close(a, b)

    def test_create_mock_attention_matrix_shape(self):
        """``create_mock_attention_matrix`` must return [B, num_heads, N, N]."""
        attn = create_mock_attention_matrix(self.B, self.num_heads, self.N)
        self.assertEqual(attn.shape, (self.B, self.num_heads, self.N, self.N))
        self.assertEqual(attn.dtype, torch.float32)

    def test_create_mock_attention_matrix_row_stochastic(self):
        """Each attention slice should sum to 1 across rows."""
        attn = create_mock_attention_matrix(self.B, self.num_heads, self.N)
        row_sums = attn.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    @unittest.skipIf(
        __import__("importlib").util.find_spec("matplotlib") is None,
        "matplotlib not installed",
    )
    def test_mock_matplotlib_figure(self):
        """``mock_matplotlib_figure`` must return a non-None figure."""
        fig = mock_matplotlib_figure()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, "savefig"))

    def test_create_mock_corpus_length(self):
        """``create_mock_corpus`` must return the requested number of sentences."""
        corpus = create_mock_corpus(num_sentences=7)
        self.assertEqual(len(corpus), 7)

    def test_create_mock_corpus_format(self):
        """Each corpus entry must be a dict with a 'text' key."""
        corpus = create_mock_corpus(num_sentences=3)
        for entry in corpus:
            self.assertIsInstance(entry, dict)
            self.assertIn("text", entry)
            self.assertIsInstance(entry["text"], str)

    def test_create_mock_corpus_default_length(self):
        """Default corpus length should be 10."""
        corpus = create_mock_corpus()
        self.assertEqual(len(corpus), 10)


if __name__ == "__main__":
    unittest.main()
