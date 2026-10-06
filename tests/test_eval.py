"""Tests for evaluation metrics in eval/__init__.py."""
from __future__ import annotations

import numpy as np

from eval import cosine_argmax, rel_frobenius, retrieval_hit_from_affinity


def test_rel_frobenius_zero_when_equal():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert rel_frobenius(a, a) == 0.0


def test_rel_frobenius_scales():
    a = np.array([[1.0, 0.0]])
    b = np.array([[2.0, 0.0]])
    # ||a-b||/||b|| = 1/2
    assert abs(rel_frobenius(a, b) - 0.5) < 1e-12


def test_rel_frobenius_zero_denominator_safe():
    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    assert rel_frobenius(a, b) == 0.0


def test_cosine_argmax_picks_best():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    query = np.array([0.9, 0.1])
    assert cosine_argmax(query, matrix) == 0


def test_retrieval_hit_argmax():
    aff = np.array([0.1, 0.9, 0.2, 0.0])
    # candidate positions are 0..T-2 (T=5 -> positions 0..3)
    assert retrieval_hit_from_affinity(aff, r=1, T=5) == 1
    assert retrieval_hit_from_affinity(aff, r=0, T=5) == 0


def test_retrieval_hit_with_neg_inf():
    aff = np.array([-np.inf, -np.inf, 5.0, -np.inf])
    # only finite candidate is index 2
    assert retrieval_hit_from_affinity(aff, r=2, T=5) == 1
    assert retrieval_hit_from_affinity(aff, r=0, T=5) == 0


def test_retrieval_hit_all_neg_inf():
    aff = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
    # no finite candidate -> cannot hit
    assert retrieval_hit_from_affinity(aff, r=0, T=5) == 0
