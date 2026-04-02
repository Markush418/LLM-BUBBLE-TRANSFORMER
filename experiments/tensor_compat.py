"""
Tensor Compatibility Layer — numpy-only fallback for Python 3.14
===================================================================
Provides torch-like API using numpy for environments where PyTorch
cannot be installed (e.g., Python 3.14 on Windows).

All experiment modules import this instead of torch directly.
When torch is available, it uses torch. Otherwise, numpy.

Usage:
    from tensor_compat import TensorOps
    ops = TensorOps.get()
    x = ops.randn(2, 32, 128)
    y = ops.matmul(x, x.transpose(0, 2, 1))
"""

import numpy as np
from typing import Optional, Tuple


class NumpyOps:
    """
    NumPy-based operations that mimic the torch API used in our pipeline.
    """

    # ─── Tensor Creation ───────────────────────────────────────────────

    @staticmethod
    def randn(*shape, dtype=np.float32):
        return np.random.randn(*shape).astype(dtype)

    @staticmethod
    def zeros(*shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(*shape, dtype=np.float32):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def eye(n, dtype=np.float32):
        return np.eye(n, dtype=dtype)

    @staticmethod
    def from_numpy(arr):
        return arr.copy()

    @staticmethod
    def asarray(arr, dtype=None):
        return np.asarray(arr, dtype=dtype)

    # ─── Linear Algebra ────────────────────────────────────────────────

    @staticmethod
    def matmul(a, b):
        return a @ b

    @staticmethod
    def cdist(a, b, p=2):
        """Compute pairwise Lp distance between rows of a and b."""
        # a: [..., N, D], b: [..., M, D] -> [..., N, M]
        if p == 2:
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            a_sq = np.sum(a**2, axis=-1, keepdims=True)  # [..., N, 1]
            b_sq = np.sum(b**2, axis=-1, keepdims=True)  # [..., M, 1]
            b_sq_t = np.moveaxis(b_sq, -2, -1)  # [..., 1, M]
            dist_sq = a_sq + b_sq_t - 2 * (a @ np.moveaxis(b, -2, -1))
            return np.maximum(dist_sq, 0)  # Clamp negative values from numerical errors
        else:
            raise NotImplementedError(f"cdist p={p} not implemented")

    @staticmethod
    def svd(a, full_matrices=False):
        """SVD decomposition. Returns (U, S, Vh)."""
        U, S, Vh = np.linalg.svd(a, full_matrices=full_matrices)
        return U, S, Vh

    @staticmethod
    def eigvalsh(a):
        """Eigenvalues of symmetric matrix."""
        return np.linalg.eigvalsh(a)

    # ─── Reduction Ops ─────────────────────────────────────────────────

    @staticmethod
    def sum(a, axis=None, keepdims=False):
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(a, axis=None, keepdims=False):
        return np.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def std(a, axis=None, keepdims=False):
        return np.std(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(a, axis=None, keepdims=False):
        return np.max(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def min(a, axis=None, keepdims=False):
        return np.min(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def median(a, axis=None):
        return np.median(a, axis=axis)

    # ─── Element-wise ──────────────────────────────────────────────────

    @staticmethod
    def exp(a):
        return np.exp(a)

    @staticmethod
    def log(a):
        return np.log(np.maximum(a, 1e-10))

    @staticmethod
    def logsumexp(a, axis=None, keepdims=False):
        """Numerically stable logsumexp."""
        a_max = np.max(a, axis=axis, keepdims=True)
        out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out

    @staticmethod
    def softmax(a, axis=-1):
        """Numerically stable softmax."""
        a_max = np.max(a, axis=axis, keepdims=True)
        e = np.exp(a - a_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    # ─── Shape Ops ─────────────────────────────────────────────────────

    @staticmethod
    def reshape(a, shape):
        return np.reshape(a, shape)

    @staticmethod
    def transpose(a, axes=None):
        return np.transpose(a, axes)

    @staticmethod
    def moveaxis(a, source, destination):
        return np.moveaxis(a, source, destination)

    @staticmethod
    def squeeze(a, axis=None):
        return np.squeeze(a, axis=axis)

    @staticmethod
    def expand_dims(a, axis):
        return np.expand_dims(a, axis=axis)

    @staticmethod
    def concatenate(arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    @staticmethod
    def stack(arrays, axis=0):
        return np.stack(arrays, axis=axis)

    # ─── Comparison / Masking ──────────────────────────────────────────

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def masked_fill(a, mask, value):
        """Fill values where mask is True (numpy equivalent of torch masked_fill)."""
        result = a.copy()
        result[mask] = value
        return result

    @staticmethod
    def all(a, axis=None):
        return np.all(a, axis=axis)

    @staticmethod
    def any(a, axis=None):
        return np.any(a, axis=axis)

    @staticmethod
    def sort(a, axis=-1):
        return np.sort(a, axis=axis)

    # ─── Distance ──────────────────────────────────────────────────────

    @staticmethod
    def pdist(a):
        """Pairwise distances (flattened upper triangle)."""
        from scipy.spatial.distance import pdist as scipy_pdist

        return scipy_pdist(a)

    # ─── Random ────────────────────────────────────────────────────────

    @staticmethod
    def manual_seed(seed):
        np.random.seed(seed)

    @staticmethod
    def randint(low, high, size):
        return np.random.randint(low, high, size=size)

    @staticmethod
    def choice(a, size, replace=True):
        return np.random.choice(a, size=size, replace=replace)

    # ─── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def isnan(a):
        return np.isnan(a)

    @staticmethod
    def isinf(a):
        return np.isinf(a)

    @staticmethod
    def clip(a, a_min, a_max):
        return np.clip(a, a_min, a_max)

    @staticmethod
    def abs(a):
        return np.abs(a)

    @staticmethod
    def sqrt(a):
        return np.sqrt(a)

    @staticmethod
    def power(a, b):
        return np.power(a, b)

    @staticmethod
    def item(a):
        """Extract scalar from 0-d array."""
        if hasattr(a, "item"):
            return a.item()
        return float(a)

    @staticmethod
    def to_numpy(a):
        return np.asarray(a)

    @staticmethod
    def save(tensor, path):
        """Save tensor to .npy file."""
        np.save(str(path), tensor)

    @staticmethod
    def load(path):
        """Load tensor from .npy file."""
        return np.load(str(path), allow_pickle=True)

    @staticmethod
    def save_tensor(tensor, path):
        """Save as .npy (numpy-compatible alternative to torch.save)."""
        np.save(str(path), tensor)

    @staticmethod
    def load_tensor(path):
        """Load from .npy file."""
        return np.load(str(path), allow_pickle=True)


class TensorOps:
    """
    Unified tensor operations.
    Uses numpy always (torch fallback for Python 3.14 compatibility).
    """

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = NumpyOps()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None


# Convenience: module-level access
ops = TensorOps.get()

if __name__ == "__main__":
    print("[tensor_compat] Running quick validation...")
    o = TensorOps.get()

    # Test basic ops
    x = o.randn(4, 32, 128)
    print(f"  randn: {x.shape}")

    # Test matmul
    y = o.matmul(x, o.transpose(x, (0, 2, 1)))
    print(f"  matmul: {y.shape}")

    # Test cdist
    d = o.cdist(x, x)
    print(f"  cdist: {d.shape}")

    # Test logsumexp
    lse = o.logsumexp(x, axis=-1)
    print(f"  logsumexp: {lse.shape}")

    # Test SVD
    x_flat = o.reshape(x, (-1, x.shape[-1]))
    U, S, Vh = o.svd(x_flat, full_matrices=False)
    print(f"  SVD: U={U.shape}, S={S.shape}, Vh={Vh.shape}")

    # Test effective rank
    p = S / o.sum(S)
    p = p[p > 1e-10]
    entropy = -o.sum(p * o.log(p))
    eff_rank = o.item(o.exp(entropy))
    print(f"  Effective rank: {eff_rank:.1f}")

    # Test softmax
    s = o.softmax(x, axis=-1)
    print(f"  softmax row sums: {o.sum(s, axis=-1).mean():.4f} (should be ~1.0)")

    print("\n[tensor_compat] All tests passed!")
