"""Protocol contracts for attention variants and the router.

Unified semantic: every variant exposes ``attend(q, k, v) -> AttentionOutput``
where the score "larger => more attention mass" (softmax-normalized along the
key axis). This keeps cross-variant evaluation fair.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.types import AttentionOutput


@runtime_checkable
class AttentionMechanism(Protocol):
    name: str

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        """Run attention for query/key/value of shape (T, d)."""
        ...

    def theoretical_complexity(self, T: int, d: int) -> str:
        """A human-readable complexity class, e.g. 'O(T^2 d)' or 'O(T d^2)'."""
        ...
