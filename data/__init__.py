"""Data generators for EffAttnForge (synthetic, seed-reproducible)."""
from __future__ import annotations

import numpy as np


def _unit_norm_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def make_retrieval(
    rng: np.random.Generator, T: int, d: int, n_samples: int
):
    """In-context associative recall.

    Builds ``n_samples`` sequences of length ``T``. Positions 0..T-2 hold a
    (key, value) store with unit-norm Gaussian keys and random values. The
    last position is a QUERY token whose key equals the key at a random store
    index ``r``. The correct output at the query position is ``value[r]``.

    Returns
    -------
    keys_seq : (n_samples, T, d)  full key sequence (store + query key)
    values_seq : (n_samples, T, d)  full value sequence (store + zero at query)
    r : (n_samples,)  ground-truth store index to retrieve
    """
    if T < 3:
        raise ValueError("retrieval needs T >= 3 (store + query)")
    store = T - 1
    keys = _unit_norm_rows(rng.standard_normal((n_samples, store, d)))
    values = rng.standard_normal((n_samples, store, d))
    r = rng.integers(0, store, size=n_samples)
    query_key = keys[np.arange(n_samples), r, :]  # (n_samples, d)
    keys_seq = np.concatenate(
        [keys, query_key[:, None, :]], axis=1
    )  # (n_samples, T, d)
    values_seq = np.concatenate(
        [values, np.zeros((n_samples, 1, d), dtype=values.dtype)], axis=1
    )
    return keys_seq, values_seq, r


def make_random_attention(
    rng: np.random.Generator, T: int, d: int, n_samples: int
):
    """Random Q/K/V for the approximation-fidelity benchmark (G1).

    Q and K are row unit-normalized, mirroring how attention is used in
    practice (post-LayerNorm), so the FAVOR+ kernel estimator and exact
    softmax are compared on the same footing.
    """
    q = _unit_norm_rows(rng.standard_normal((n_samples, T, d)))
    k = _unit_norm_rows(rng.standard_normal((n_samples, T, d)))
    v = rng.standard_normal((n_samples, T, d))
    return q, k, v
