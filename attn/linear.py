"""Linear attention (Katharopoulos et al. 2020) — O(T d^2), no T^2 matrix."""
from __future__ import annotations

import time

import numpy as np

from core.types import AttentionOutput


def _mb(nbytes: float) -> float:
    return nbytes / (1024.0 * 1024.0)


def _elu(x: np.ndarray) -> np.ndarray:
    """elu(x) = x if x>=0 else exp(x)-1 (numpy has no np.elu)."""
    return np.where(x >= 0.0, x, np.expm1(x))


def _phi(x: np.ndarray) -> np.ndarray:
    """Feature map phi(x) = elu(x) + 1 (strictly positive)."""
    return _elu(x) + 1.0


class LinearAttention:
    """Kernelized linear attention. Bidirectional (non-causal).

    out_t = (phi(q_t)^T S) / (phi(q_t)^T z),  S = sum_s phi(k_s) v_s^T,
    z = sum_s phi(k_s).  Complexity O(T d^2), memory O(d^2).
    """

    name = "linear"

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        t0 = time.perf_counter()
        kf = _phi(k)  # (T, d)
        qf = _phi(q)  # (T, d)
        S = kf.T @ v  # (d, d)
        z = kf.sum(axis=0)  # (d,)
        num = qf @ S  # (T, d)
        den = qf @ z  # (T,)
        safe = np.where(np.abs(den) < 1e-12, 1.0, den)
        out = num / safe[:, None]
        cost = time.perf_counter() - t0
        mem = _mb(2 * T * d * 8 + d * d * 8)
        return AttentionOutput(out, None, cost, mem, self.name)

    def query_affinity(self, q_last: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Raw kernel affinity phi(q_last) . phi(k_j) for every key j."""
        return _phi(q_last) @ _phi(k).T

    def theoretical_complexity(self, T: int, d: int) -> str:
        return "O(T d^2)"
