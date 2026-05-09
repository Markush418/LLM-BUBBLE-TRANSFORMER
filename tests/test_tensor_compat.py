"""Tests for experiments/tensor_compat.py — NumPy-based PyTorch API fallback.

These tests verify that NumpyOps correctly mimics the torch API using only
NumPy, and that TensorOps.get() returns a usable singleton without importing
actual PyTorch.
"""

import sys
import os
import unittest
import tempfile
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import numpy as np
from tensor_compat import NumpyOps, TensorOps


class TestNumpyOpsTensorCreation(unittest.TestCase):
    """Tests for tensor creation static methods."""

    def setUp(self):
        self.ops = NumpyOps()

    def test_randn_shape_and_dtype(self):
        x = self.ops.randn(2, 32, 128)
        self.assertEqual(x.shape, (2, 32, 128))
        self.assertEqual(x.dtype, np.float32)

    def test_zeros_shape_and_dtype(self):
        x = self.ops.zeros(3, 4, dtype=np.float64)
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(x.dtype, np.float64)
        self.assertTrue(np.all(x == 0))

    def test_ones_shape_and_dtype(self):
        x = self.ops.ones(5, 5)
        self.assertEqual(x.shape, (5, 5))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.all(x == 1))

    def test_eye(self):
        x = self.ops.eye(4)
        self.assertEqual(x.shape, (4, 4))
        self.assertTrue(np.allclose(x, np.eye(4, dtype=np.float32)))

    def test_from_numpy_copies(self):
        arr = np.array([1.0, 2.0, 3.0])
        x = self.ops.from_numpy(arr)
        self.assertTrue(np.array_equal(x, arr))
        x[0] = 99.0
        self.assertEqual(arr[0], 1.0)

    def test_asarray(self):
        arr = [1.0, 2.0, 3.0]
        x = self.ops.asarray(arr, dtype=np.float32)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.dtype, np.float32)


class TestNumpyOpsLinearAlgebra(unittest.TestCase):
    """Tests for linear algebra static methods."""

    def setUp(self):
        self.ops = NumpyOps()

    def test_matmul(self):
        a = np.ones((3, 4), dtype=np.float32)
        b = np.ones((4, 5), dtype=np.float32)
        c = self.ops.matmul(a, b)
        self.assertEqual(c.shape, (3, 5))
        self.assertTrue(np.allclose(c, 4.0))

    def test_cdist_shape_and_non_negative(self):
        a = np.random.randn(10, 128).astype(np.float32)
        b = np.random.randn(8, 128).astype(np.float32)
        d = self.ops.cdist(a, b, p=2)
        self.assertEqual(d.shape, (10, 8))
        self.assertTrue(np.all(d >= 0))

    def test_cdist_unsupported_p(self):
        with self.assertRaises(NotImplementedError):
            self.ops.cdist(np.zeros((2, 2)), np.zeros((2, 2)), p=1)

    def test_svd(self):
        a = np.random.randn(4, 6).astype(np.float32)
        U, S, Vh = self.ops.svd(a, full_matrices=False)
        self.assertEqual(U.shape, (4, 4))
        self.assertEqual(S.shape, (4,))
        self.assertEqual(Vh.shape, (4, 6))
        recon = U @ np.diag(S) @ Vh
        self.assertTrue(np.allclose(recon, a, atol=1e-5))

    def test_eigvalsh(self):
        a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        vals = self.ops.eigvalsh(a)
        self.assertEqual(vals.shape, (2,))
        self.assertTrue(np.allclose(np.sort(vals), np.array([1.0, 3.0])))


class TestNumpyOpsReductionAndElementWise(unittest.TestCase):
    """Tests for reduction and element-wise operations."""

    def setUp(self):
        self.ops = NumpyOps()
        self.arr = np.arange(12).reshape(3, 4).astype(np.float32)

    def test_sum(self):
        self.assertEqual(self.ops.sum(self.arr), 66.0)
        axis_sum = self.ops.sum(self.arr, axis=1)
        self.assertEqual(axis_sum.shape, (3,))

    def test_mean(self):
        self.assertEqual(self.ops.mean(self.arr), 5.5)

    def test_std(self):
        self.assertGreater(self.ops.std(self.arr), 0)

    def test_max(self):
        self.assertEqual(self.ops.max(self.arr), 11.0)

    def test_min(self):
        self.assertEqual(self.ops.min(self.arr), 0.0)

    def test_median(self):
        self.assertEqual(self.ops.median(self.arr), 5.5)

    def test_exp(self):
        x = np.array([0.0, 1.0], dtype=np.float32)
        y = self.ops.exp(x)
        self.assertTrue(np.allclose(y, np.array([1.0, np.e])))

    def test_log_clamps_small_values(self):
        x = np.array([1e-12, 1.0], dtype=np.float32)
        y = self.ops.log(x)
        self.assertFalse(np.any(np.isinf(y)))
        self.assertFalse(np.any(np.isnan(y)))

    def test_logsumexp(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        lse = self.ops.logsumexp(x)
        expected = np.log(np.sum(np.exp(x)))
        self.assertTrue(np.allclose(lse, expected, atol=1e-5))

    def test_logsumexp_keepdims(self):
        x = np.ones((2, 3), dtype=np.float32)
        lse = self.ops.logsumexp(x, axis=1, keepdims=True)
        self.assertEqual(lse.shape, (2, 1))

    def test_softmax_sums_to_one(self):
        x = np.random.randn(4, 8).astype(np.float32)
        s = self.ops.softmax(x, axis=-1)
        sums = s.sum(axis=-1)
        self.assertTrue(np.allclose(sums, 1.0))


class TestNumpyOpsShapeOps(unittest.TestCase):
    """Tests for shape manipulation operations."""

    def setUp(self):
        self.ops = NumpyOps()
        self.arr = np.arange(24).reshape(2, 3, 4).astype(np.float32)

    def test_reshape(self):
        y = self.ops.reshape(self.arr, (6, 4))
        self.assertEqual(y.shape, (6, 4))

    def test_transpose(self):
        y = self.ops.transpose(self.arr, (2, 0, 1))
        self.assertEqual(y.shape, (4, 2, 3))

    def test_moveaxis(self):
        y = self.ops.moveaxis(self.arr, 0, 2)
        self.assertEqual(y.shape, (3, 4, 2))

    def test_squeeze(self):
        x = np.zeros((2, 1, 4, 1), dtype=np.float32)
        y = self.ops.squeeze(x, axis=1)
        self.assertEqual(y.shape, (2, 4, 1))

    def test_expand_dims(self):
        x = np.zeros((2, 3), dtype=np.float32)
        y = self.ops.expand_dims(x, axis=1)
        self.assertEqual(y.shape, (2, 1, 3))

    def test_concatenate(self):
        a = np.zeros((2, 3), dtype=np.float32)
        b = np.ones((2, 3), dtype=np.float32)
        c = self.ops.concatenate([a, b], axis=0)
        self.assertEqual(c.shape, (4, 3))

    def test_stack(self):
        a = np.zeros((2, 3), dtype=np.float32)
        b = np.ones((2, 3), dtype=np.float32)
        c = self.ops.stack([a, b], axis=0)
        self.assertEqual(c.shape, (2, 2, 3))


class TestNumpyOpsComparisonAndMasking(unittest.TestCase):
    """Tests for comparison, masking, and utility operations."""

    def setUp(self):
        self.ops = NumpyOps()

    def test_where(self):
        cond = np.array([True, False, True])
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        z = self.ops.where(cond, x, y)
        self.assertTrue(np.array_equal(z, np.array([1.0, 20.0, 3.0])))

    def test_masked_fill(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([False, True, False, True])
        z = self.ops.masked_fill(a, mask, 99.0)
        self.assertTrue(np.array_equal(z, np.array([1.0, 99.0, 3.0, 99.0])))
        self.assertEqual(a[1], 2.0)

    def test_all(self):
        a = np.array([True, True, False])
        self.assertFalse(self.ops.all(a))
        self.assertTrue(self.ops.all(np.array([True, True])))

    def test_any(self):
        a = np.array([False, False, True])
        self.assertTrue(self.ops.any(a))
        self.assertFalse(self.ops.any(np.array([False, False])))

    def test_sort(self):
        a = np.array([3.0, 1.0, 2.0])
        s = self.ops.sort(a)
        self.assertTrue(np.array_equal(s, np.array([1.0, 2.0, 3.0])))

    def test_isnan(self):
        a = np.array([1.0, np.nan, 3.0])
        self.assertTrue(
            np.array_equal(self.ops.isnan(a), np.array([False, True, False]))
        )

    def test_isinf(self):
        a = np.array([1.0, np.inf, 3.0])
        self.assertTrue(
            np.array_equal(self.ops.isinf(a), np.array([False, True, False]))
        )

    def test_clip(self):
        a = np.array([-1.0, 0.5, 2.0, 5.0])
        c = self.ops.clip(a, 0.0, 1.0)
        self.assertTrue(np.array_equal(c, np.array([0.0, 0.5, 1.0, 1.0])))

    def test_abs(self):
        a = np.array([-1.0, 2.0, -3.0])
        self.assertTrue(np.array_equal(self.ops.abs(a), np.array([1.0, 2.0, 3.0])))

    def test_sqrt(self):
        a = np.array([4.0, 9.0, 16.0])
        self.assertTrue(np.allclose(self.ops.sqrt(a), np.array([2.0, 3.0, 4.0])))

    def test_power(self):
        a = np.array([2.0, 3.0])
        self.assertTrue(np.allclose(self.ops.power(a, 2.0), np.array([4.0, 9.0])))

    def test_item(self):
        a = np.array(42.0)
        self.assertEqual(self.ops.item(a), 42.0)
        self.assertEqual(self.ops.item(7.0), 7.0)

    def test_to_numpy(self):
        a = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(self.ops.to_numpy(a), a))


class TestNumpyOpsPdist(unittest.TestCase):
    """Tests for pdist (requires scipy)."""

    def setUp(self):
        self.ops = NumpyOps()

    @unittest.skipIf(
        importlib.util.find_spec("scipy") is None,
        "scipy not installed",
    )
    def test_pdist_shape(self):
        a = np.random.randn(5, 10).astype(np.float32)
        d = self.ops.pdist(a)
        expected_len = 5 * (5 - 1) // 2
        self.assertEqual(d.shape, (expected_len,))

    @unittest.skipIf(
        importlib.util.find_spec("scipy") is None,
        "scipy not installed",
    )
    def test_pdist_non_negative(self):
        a = np.random.randn(5, 10).astype(np.float32)
        d = self.ops.pdist(a)
        self.assertTrue(np.all(d >= 0))

    def test_pdist_scipy_not_installed_raises(self):
        """If scipy is unavailable, pdist should raise ImportError."""
        if importlib.util.find_spec("scipy") is not None:
            self.skipTest("scipy is installed")
        with self.assertRaises(ImportError):
            self.ops.pdist(np.zeros((2, 2), dtype=np.float32))


class TestNumpyOpsRandom(unittest.TestCase):
    """Tests for random operations."""

    def setUp(self):
        self.ops = NumpyOps()

    def test_manual_seed_reproducibility(self):
        self.ops.manual_seed(42)
        a = np.random.randn(5)
        self.ops.manual_seed(42)
        b = np.random.randn(5)
        self.assertTrue(np.allclose(a, b))

    def test_randint(self):
        r = self.ops.randint(0, 10, size=(20,))
        self.assertEqual(r.shape, (20,))
        self.assertTrue(np.all((r >= 0) & (r < 10)))

    def test_choice(self):
        items = np.arange(100)
        chosen = self.ops.choice(items, size=10, replace=False)
        self.assertEqual(chosen.shape, (10,))
        self.assertEqual(len(np.unique(chosen)), 10)


class TestNumpyOpsSaveLoad(unittest.TestCase):
    """Tests for save/load tensor operations."""

    def setUp(self):
        self.ops = NumpyOps()

    def test_save_and_load(self):
        arr = np.random.randn(3, 4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npy")
            self.ops.save(arr, path)
            loaded = self.ops.load(path)
            self.assertTrue(np.allclose(arr, loaded))

    def test_save_tensor_and_load_tensor(self):
        arr = np.random.randn(2, 5).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npy")
            self.ops.save_tensor(arr, path)
            loaded = self.ops.load_tensor(path)
            self.assertTrue(np.allclose(arr, loaded))


class TestTensorOpsSingleton(unittest.TestCase):
    """Tests for TensorOps singleton behavior."""

    def setUp(self):
        TensorOps.reset()

    def tearDown(self):
        TensorOps.reset()

    def test_get_returns_instance(self):
        ops = TensorOps.get()
        self.assertIsInstance(ops, NumpyOps)

    def test_get_same_instance(self):
        ops1 = TensorOps.get()
        ops2 = TensorOps.get()
        self.assertIs(ops1, ops2)

    def test_reset_creates_new_instance(self):
        ops1 = TensorOps.get()
        TensorOps.reset()
        ops2 = TensorOps.get()
        self.assertIsNot(ops1, ops2)
        self.assertIsInstance(ops2, NumpyOps)


class TestNoPyTorchImport(unittest.TestCase):
    """Verify that tensor_compat can be imported and used without PyTorch."""

    def test_torch_not_in_tensor_compat(self):
        """Ensure tensor_compat module itself does not import torch."""
        import ast

        path = os.path.join(
            os.path.dirname(__file__), "..", "experiments", "tensor_compat.py"
        )
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
        import_froms = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        for node in imports:
            for alias in node.names:
                self.assertNotEqual(alias.name, "torch")
        for node in import_froms:
            self.assertNotEqual(node.module, "torch")

    def test_runs_without_torch(self):
        """TensorOps.get() should return NumpyOps even when torch is unavailable."""
        ops = TensorOps.get()
        x = ops.randn(2, 3)
        self.assertIsInstance(x, np.ndarray)
        y = ops.sum(x)
        self.assertIsInstance(y, (np.ndarray, np.generic, float))


if __name__ == "__main__":
    unittest.main()
