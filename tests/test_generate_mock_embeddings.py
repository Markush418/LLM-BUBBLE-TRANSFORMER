"""Unit tests for the synthetic/mock embedding generator.

These tests verify that ``generate_mock_embeddings.py`` produces
embeddings with the correct shapes, dtypes, reproducibility, and
on-disk format — all without requiring a GPU or a real model.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import numpy as np

from generate_mock_embeddings import (
    generate_layer_embeddings,
    generate_raw_input_embeddings,
    generate_attention_mask,
    _quick_effective_rank,
    save_mock_embeddings,
)


class TestGenerateMockEmbeddings(unittest.TestCase):
    """Test suite for mock embedding generation functions."""

    def setUp(self):
        """Standard small dimensions for fast, deterministic tests."""
        self.num_layers = 6
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 64
        self.num_heads = 4
        self.seed = 42

    # ------------------------------------------------------------------
    # 1. Layer embeddings
    # ------------------------------------------------------------------

    def test_generate_layer_embeddings_shape(self):
        """Output must be a dict with [B, N, D] arrays for every layer."""
        emb = generate_layer_embeddings(
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            seed=self.seed,
        )
        self.assertIsInstance(emb, dict)
        self.assertEqual(len(emb), self.num_layers)
        for layer_idx in range(self.num_layers):
            self.assertIn(layer_idx, emb)
            arr = emb[layer_idx]
            self.assertEqual(arr.shape, (self.batch_size, self.seq_len, self.d_model))
            self.assertEqual(arr.dtype, np.float32)

    def test_generate_layer_embeddings_reproducibility(self):
        """Same seed must yield identical embeddings."""
        emb_a = generate_layer_embeddings(
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            seed=self.seed,
        )
        emb_b = generate_layer_embeddings(
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            seed=self.seed,
        )
        for layer_idx in range(self.num_layers):
            np.testing.assert_array_equal(emb_a[layer_idx], emb_b[layer_idx])

    def test_generate_layer_embeddings_unit_norm(self):
        """Every token vector should be L2-normalised to ~1.0."""
        emb = generate_layer_embeddings(
            num_layers=3,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            seed=self.seed,
        )
        for arr in emb.values():
            flat = arr.reshape(-1, arr.shape[-1])
            norms = np.linalg.norm(flat, axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_layer_geometry_progression(self):
        """Later layers should be more anisotropic (lower effective rank)."""
        emb = generate_layer_embeddings(
            num_layers=8,
            batch_size=4,
            seq_len=32,
            d_model=128,
            num_heads=4,
            seed=self.seed,
        )
        ranks = [_quick_effective_rank(emb[i]) for i in range(8)]
        # Early layer should have higher effective rank than late layer
        self.assertGreater(ranks[0], ranks[-1])

    # ------------------------------------------------------------------
    # 2. Raw input embeddings
    # ------------------------------------------------------------------

    def test_generate_raw_input_embeddings_shape(self):
        """Must return [B, N, D] float32 array."""
        raw = generate_raw_input_embeddings(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            seed=self.seed,
        )
        self.assertEqual(raw.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(raw.dtype, np.float32)

    def test_generate_raw_input_embeddings_unit_norm(self):
        """Vocab-derived embeddings should be L2-normalised."""
        raw = generate_raw_input_embeddings(
            batch_size=4,
            seq_len=32,
            d_model=128,
            seed=self.seed,
        )
        flat = raw.reshape(-1, raw.shape[-1])
        norms = np.linalg.norm(flat, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # ------------------------------------------------------------------
    # 3. Attention mask
    # ------------------------------------------------------------------

    def test_generate_attention_mask_shape_and_values(self):
        """Must be [B, N] int64 array with only 0/1 values."""
        mask = generate_attention_mask(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            seed=self.seed,
        )
        self.assertEqual(mask.shape, (self.batch_size, self.seq_len))
        self.assertEqual(mask.dtype, np.int64)
        self.assertTrue(np.all(np.isin(mask, [0, 1])))

    def test_generate_attention_mask_reproducibility(self):
        """Same seed must yield the identical mask."""
        mask_a = generate_attention_mask(
            batch_size=self.batch_size, seq_len=self.seq_len, seed=self.seed
        )
        mask_b = generate_attention_mask(
            batch_size=self.batch_size, seq_len=self.seq_len, seed=self.seed
        )
        np.testing.assert_array_equal(mask_a, mask_b)

    # ------------------------------------------------------------------
    # 4. Effective rank helper
    # ------------------------------------------------------------------

    def test_quick_effective_rank_sanity(self):
        """Rank of a perfectly isotropic cloud should be close to D."""
        rng = np.random.RandomState(self.seed)
        # Isotropic Gaussian in 50-D
        cloud = rng.randn(1000, 50).astype(np.float32)
        rank = _quick_effective_rank(cloud)
        self.assertGreater(rank, 40.0)
        self.assertLess(rank, 50.0)

    def test_quick_effective_rank_low_rank(self):
        """Rank of a low-rank cloud should be close to the true rank."""
        rng = np.random.RandomState(self.seed)
        true_rank = 5
        u = rng.randn(200, true_rank).astype(np.float32)
        v = rng.randn(true_rank, 64).astype(np.float32)
        cloud = u @ v
        rank = _quick_effective_rank(cloud)
        self.assertGreater(rank, true_rank - 2)
        self.assertLess(rank, true_rank + 2)

    # ------------------------------------------------------------------
    # 5. Save / on-disk format
    # ------------------------------------------------------------------

    def test_save_mock_embeddings_creates_expected_files(self):
        """All expected files must be written with correct shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_mock_embeddings(
                output_dir=tmpdir,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                d_model=self.d_model,
                num_heads=self.num_heads,
                seed=self.seed,
            )
            # Layer files
            for layer_idx in range(self.num_layers):
                path = os.path.join(tmpdir, "softmax", f"layer_{layer_idx}.npy")
                self.assertTrue(os.path.isfile(path))
                arr = np.load(path)
                self.assertEqual(
                    arr.shape, (self.batch_size, self.seq_len, self.d_model)
                )

            # Raw input
            raw_path = os.path.join(tmpdir, "raw_input.npy")
            self.assertTrue(os.path.isfile(raw_path))
            raw = np.load(raw_path)
            self.assertEqual(raw.shape, (self.batch_size, self.seq_len, self.d_model))

            # Attention mask
            mask_path = os.path.join(tmpdir, "attention_mask.npy")
            self.assertTrue(os.path.isfile(mask_path))
            mask = np.load(mask_path)
            self.assertEqual(mask.shape, (self.batch_size, self.seq_len))

            # Metadata
            meta_path = os.path.join(tmpdir, "metadata.json")
            self.assertTrue(os.path.isfile(meta_path))

    def test_metadata_matches_real_extraction_format(self):
        """Metadata JSON must contain the keys needed by downstream consumers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_mock_embeddings(
                output_dir=tmpdir,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                d_model=self.d_model,
                num_heads=self.num_heads,
                seed=self.seed,
            )
            meta_path = os.path.join(tmpdir, "metadata.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.assertIn("mode", meta)
            self.assertEqual(meta["mode"], "mock")
            self.assertIn("num_layers", meta)
            self.assertIn("batch_size", meta)
            self.assertIn("seq_len", meta)
            self.assertIn("d_model", meta)
            self.assertIn("num_heads", meta)
            self.assertIn("seed", meta)
            self.assertIn("description", meta)

    # ------------------------------------------------------------------
    # 6. Various dimensions
    # ------------------------------------------------------------------

    def test_various_dimensions(self):
        """Generator should work with non-default dimensions."""
        configs = [
            {"batch_size": 1, "seq_len": 8, "d_model": 32},
            {"batch_size": 8, "seq_len": 128, "d_model": 256},
            {"batch_size": 2, "seq_len": 64, "d_model": 1024},
        ]
        for cfg in configs:
            with self.subTest(**cfg):
                emb = generate_layer_embeddings(
                    num_layers=2,
                    num_heads=2,
                    seed=self.seed,
                    **cfg,
                )
                self.assertEqual(
                    emb[0].shape,
                    (cfg["batch_size"], cfg["seq_len"], cfg["d_model"]),
                )


if __name__ == "__main__":
    unittest.main()
