"""Evaluation metrics for EffAttnForge."""
from __future__ import annotations

import numpy as np


def rel_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b||_F / ||b||_F (0 when b is ~zero)."""
    denom = float(np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def cosine_argmax(query: np.ndarray, matrix: np.ndarray) -> int:
    """Index of the row in ``matrix`` most cosine-similar to ``query``."""
    qn = float(np.linalg.norm(query)) + 1e-12
    mn = np.linalg.norm(matrix, axis=-1) + 1e-12
    sims = (matrix @ query) / (qn * mn)
    return int(np.argmax(sims))


def retrieval_hit_from_affinity(affinity_last: np.ndarray, r: int, T: int) -> int:
    """1 if the matching key r has the highest affinity among store positions.

    Candidates are store positions 0..T-2 (the query sits at T-1 and is
    excluded). Sparse variants return -inf for unattended keys, so a far
    matching key correctly fails to win.
    """
    cand = affinity_last[: T - 1]
    if not np.all(np.isfinite(cand)):
        # tie-break: ignore -inf, argmax over finite candidates
        finite = np.isfinite(cand)
        if not finite.any():
            return 0
        restricted = np.where(finite, cand, -np.inf)
        return 1 if int(np.argmax(restricted)) == int(r) else 0
    return 1 if int(np.argmax(cand)) == int(r) else 0
