"""Shared dataclasses and result containers for EffAttnForge."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttentionOutput:
    """Result of a single attention pass.

    Interface contract: ``output`` is always (T, d). ``weights`` is the
    (T, T) normalized attention matrix when the variant materializes it
    (full attention) and ``None`` otherwise (kernel / sparse variants that
    never form the dense matrix). ``cost`` is measured wall-clock seconds for
    THIS request (batch-of-1); ``mem_peak_mb`` is the peak additional memory
    in megabytes. ``variant`` records which implementation produced it.
    """

    output: np.ndarray
    weights: np.ndarray | None
    cost: float
    mem_peak_mb: float
    variant: str


@dataclass
class VariantReport:
    """Per-variant scores for one benchmark cell (a fixed T)."""

    variant: str
    T: int
    # G1: relative Frobenius error of normalized attention vs full attention.
    approx_frobenius_rel: float
    # G2: retrieval top-1 accuracy over the in-context query task.
    retrieval_acc: float
    # G3: measured wall-clock (s) and peak memory (MB) for one forward pass.
    cost: float
    mem_peak_mb: float
    # Whether the variant was actually run (False => skipped, e.g. OOM).
    ran: bool = True
    skip_reason: str = ""
    # Cell kind: "acc" carries fidelity (G1) + retrieval (G2); "cost" carries
    # wall-clock/memory (G3). Splitting the two sweeps lets the cost sweep use
    # much larger T (where O(T^2) vs O(T) dominates) without paying the full
    # quadratic cost inside the accuracy sweep.
    kind: str = "acc"


@dataclass
class BenchmarkReport:
    """Aggregated benchmark output (one per seed)."""

    seed: int
    cells: list[VariantReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "cells": [vars(c) for c in self.cells],
        }
