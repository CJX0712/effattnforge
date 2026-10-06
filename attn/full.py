"""Exact full (softmax) attention — the SOTA reference / upper bound."""
from __future__ import annotations

import math
import time

import numpy as np

from core.types import AttentionOutput


def _mb(nbytes: float) -> float:
    return nbytes / (1024.0 * 1024.0)


class FullAttention:
    """O(T^2 d) exact softmax attention. The benchmark reference baseline."""

    name = "full"

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        t0 = time.perf_counter()
        scale = 1.0 / math.sqrt(d)
        scores = (q @ k.T) * scale  # (T, T)
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        a = e / e.sum(axis=-1, keepdims=True)  # (T, T) normalized
        out = a @ v  # (T, d)
        cost = time.perf_counter() - t0
        # Peak additional allocation: scores + normalized weights + output.
        mem = _mb(2 * T * T * 8 + T * d * 8)
        return AttentionOutput(out, a, cost, mem, self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        return "O(T^2 d)"

    def query_affinity(self, q_last: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Scaled dot-product logits of the last query against every key."""
        return (q_last @ k.T) / math.sqrt(k.shape[1])
