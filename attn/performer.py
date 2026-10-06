"""Performer / FAVOR+ (Choromanski et al. 2020) random-feature attention.

Efficient O(T d m) kernel attention. No T x T matrix is ever materialized:
    KV   = Phi(K)^T @ V          (2m, d)
    z    = Phi(K).sum(0)         (2m,)
    out  = Phi(Q) @ KV / (Phi(Q) @ z)[:, None]
The random feature map is the positive FAVOR+ estimator; W,b are drawn from a
fixed salt so two runs with the same base seed are byte-identical (gate G4).
"""
from __future__ import annotations

import time

import numpy as np

from core.seed import rng_for
from core.types import AttentionOutput


def _mb(nbytes: float) -> float:
    return nbytes / (1024.0 * 1024.0)


class PerformerAttention:
    name = "performer"

    def __init__(self, m: int = 256, salt: int = 7777) -> None:
        self.m = int(m)
        self.salt = int(salt)

    def _features(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        norm = np.exp(-0.5 * np.sum(x * x, axis=-1))  # (T,)
        proj = x @ W.T + b  # (T, m)
        feats = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)  # (T, 2m)
        scale = norm / np.sqrt(self.m)
        return scale[:, None] * feats  # (T, 2m)

    def _sample(self, d: int):
        rng = rng_for(self.salt)
        # Variance d^{-1/2} per component => E[cos(w.(x-y))] = exp(-||x-y||^2/(2 sqrt(d))),
        # matching the softmax temperature 1/sqrt(d) (Performer FAVOR+).
        W = rng.standard_normal((self.m, d)) / (d ** 0.25)
        b = rng.uniform(0.0, 2.0 * np.pi, size=self.m)
        return W, b

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T, d = q.shape
        W, b = self._sample(d)
        t0 = time.perf_counter()
        phi_q = self._features(q, W, b)  # (T, 2m)
        phi_k = self._features(k, W, b)  # (T, 2m)
        kv = phi_k.T @ v  # (2m, d)
        z = phi_k.sum(axis=0)  # (2m,)
        num = phi_q @ kv  # (T, d)
        den = phi_q @ z  # (T,)
        safe = np.where(np.abs(den) < 1e-12, 1.0, den)
        out = num / safe[:, None]
        cost = time.perf_counter() - t0
        mem = _mb(2 * T * (2 * self.m) * 8 + (2 * self.m) * d * 8)
        return AttentionOutput(out, None, cost, mem, self.name)

    def theoretical_complexity(self, T: int, d: int) -> str:
        return f"O(T d m)  [m={self.m}]"

    def query_affinity(self, q_last: np.ndarray, k: np.ndarray) -> np.ndarray:
        """FAVOR+ kernel affinity phi(q_last) . phi(k_j) for every key j."""
        W, b = self._sample(k.shape[1])
        phi_q = self._features(q_last[None, :], W, b)[0]  # (2m,)
        phi_k = self._features(k, W, b)  # (T, 2m)
        return phi_q @ phi_k.T  # (T,)
