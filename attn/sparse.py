"""Sparse attention variants (local-window, block-sparse + global).

Both are implemented WITHOUT materializing a T x T matrix, so their measured
cost reflects the true sparse complexity. On long-range in-context retrieval
these degrade honestly (the query token sits at the end and the needed key can
be far away) — that is the point of benchmarking them next to global variants.
"""
from __future__ import annotations

import math
import time

import numpy as np

from core.types import AttentionOutput


def _mb(nbytes: float) -> float:
    return nbytes / (1024.0 * 1024.0)


class LocalWindowAttention:
    """Each query attends only to positions [i-w, i+w] (clamped). O(T w d)."""

    name = "local_window"

    def __init__(self, window: int = 32) -> None:
        self.window = int(window)

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        w = self.window
        out = np.empty((T, d), dtype=np.float64)
        t0 = time.perf_counter()
        scale = 1.0 / math.sqrt(d)
        for i in range(T):
            lo = max(0, i - w)
            hi = min(T, i + w + 1)
            kk = k[lo:hi]  # (L, d)
            vv = v[lo:hi]  # (L, d)
            scores = (q[i] @ kk.T) * scale  # (L,)
            scores = scores - scores.max()
            e = np.exp(scores)
            a = e / e.sum()
            out[i] = a @ vv
        cost = time.perf_counter() - t0
        mem = _mb((2 * w + 1) * d * 8)
        return AttentionOutput(out, None, cost, mem, self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        return f"O(T w d)  [w={self.window}]"

    def query_affinity(self, q_last: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Affinity of the LAST query against keys; -inf outside the window."""
        T = k.shape[0]
        w = self.window
        aff = np.full(T, -np.inf)
        i = T - 1
        lo = max(0, i - w)
        hi = min(T, i + w + 1)
        aff[lo:hi] = (q_last @ k[lo:hi].T) / math.sqrt(k.shape[1])
        return aff


class BlockSparseAttention:
    """Block-local attention (same / prev / next block) + global tokens. O(T b d)."""

    name = "block_sparse"

    def __init__(self, block_size: int = 64, n_global: int = 8) -> None:
        self.block_size = int(block_size)
        self.n_global = int(n_global)

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        b = self.block_size
        ng = min(self.n_global, T)
        n_blocks = max(1, (T + b - 1) // b)
        out = np.empty((T, d), dtype=np.float64)
        t0 = time.perf_counter()
        scale = 1.0 / math.sqrt(d)
        global_idx = np.arange(ng)
        for bi in range(n_blocks):
            qs = bi * b
            qe = min(T, qs + b)
            # allowed key blocks: previous, same, next
            blocks = [bi - 1, bi, bi + 1]
            key_blocks = []
            for bj in blocks:
                if 0 <= bj < n_blocks:
                    ks = bj * b
                    ke = min(T, ks + b)
                    key_blocks.append((ks, ke))
            # Build allowed key index set (sorted, unique).
            idx_sets = [global_idx]
            for ks, ke in key_blocks:
                idx_sets.append(np.arange(ks, ke))
            allowed = np.unique(np.concatenate(idx_sets))
            kk = k[allowed]  # (L, d)
            vv = v[allowed]  # (L, d)
            scores = (q[qs:qe] @ kk.T) * scale  # (B, L)
            scores = scores - scores.max(axis=-1, keepdims=True)
            e = np.exp(scores)
            a = e / e.sum(axis=-1, keepdims=True)
            out[qs:qe] = a @ vv
        cost = time.perf_counter() - t0
        mem = _mb((3 * b + ng) * d * 8)
        return AttentionOutput(out, None, cost, mem, self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        return f"O(T b d)  [b={self.block_size}, g={self.n_global}]"

    def query_affinity(self, q_last: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Affinity of the LAST query against keys; -inf outside the block pattern."""
        T, d = k.shape
        b = self.block_size
        ng = min(self.n_global, T)
        n_blocks = max(1, (T + b - 1) // b)
        bi = n_blocks - 1  # last query block
        aff = np.full(T, -np.inf)
        scale = 1.0 / math.sqrt(d)
        global_idx = np.arange(ng)
        blocks = [bi - 1, bi, bi + 1]
        idx_sets = [global_idx]
        for bj in blocks:
            if 0 <= bj < n_blocks:
                ks = bj * b
                ke = min(T, ks + b)
                idx_sets.append(np.arange(ks, ke))
        allowed = np.unique(np.concatenate(idx_sets))
        aff[allowed] = (q_last @ k[allowed].T) * scale
        return aff
