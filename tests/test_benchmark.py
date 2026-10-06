"""Determinism (gate G4) and pipeline gate-structure tests.

These use a SMALL config so they stay fast in CI; they exercise the same code
paths (shared reference, split accuracy/cost sweeps, router, aggregation) as the
full benchmark that produces the S-grade result in examples/run_demo.py.
"""
from __future__ import annotations

import numpy as np

from core.config import Config
from eval.benchmark import (
    aggregate_cost,
    aggregate_fidelity,
    aggregate_retrieval,
    run_benchmark,
)
from pipeline import EffAttnPipeline


def _small_cfg():
    return Config(
        max_len=512,
        full_threshold=32,
        performer_m_short=16,
        performer_m_long=32,
        retrieval_d=16,
        retrieval_n_samples=8,
        retrieval_t_values=(64, 128),
        fidelity_d=16,
        fidelity_n_samples=4,
        cost_t_values=(128, 256),
        n_seeds=3,
    )


def test_benchmark_emits_split_cell_kinds():
    rep = run_benchmark(_small_cfg(), 42)
    kinds = {c.kind for c in rep.cells}
    assert kinds == {"acc", "cost"}
    # accuracy cells carry retrieval/fidelity, cost cells carry cost
    for c in rep.cells:
        if c.kind == "acc":
            assert c.cost == 0.0
            assert np.isfinite(c.retrieval_acc)
        else:
            assert c.kind == "cost"
            assert c.approx_frobenius_rel == 0.0
            assert c.retrieval_acc == 0.0
            assert c.cost >= 0.0


def test_aggregates_respect_cell_kind():
    rep = run_benchmark(_small_cfg(), 42)
    reports = [rep]
    # retrieval/fidelity come only from acc cells
    r = aggregate_retrieval(reports, "full")
    f = aggregate_fidelity(reports, "full")
    assert np.isfinite(r) and 0.0 <= r <= 1.0
    assert np.isfinite(f)
    # cost comes only from cost cells and sums across T (non-negative)
    c = aggregate_cost(reports, "full")
    assert c >= 0.0
    c_perf = aggregate_cost(reports, "performer")
    assert c_perf >= 0.0
    # At these small T performer is overhead-bound (NOT cheaper than full); the
    # real cost gate only flips in favour of Performer at large T (see demo).
    # Here we only assert the aggregation is well-formed.
    assert np.isfinite(c) and np.isfinite(c_perf)


def test_determinism_bit_identical(seed_a=42, seed_b=42):
    cfg = _small_cfg()
    rep_a = run_benchmark(cfg, seed_a)
    rep_b = run_benchmark(cfg, seed_b)
    cores_a = {
        (c.variant, c.T, c.kind): (c.retrieval_acc, c.approx_frobenius_rel)
        for c in rep_a.cells
        if c.kind == "acc"
    }
    cores_b = {
        (c.variant, c.T, c.kind): (c.retrieval_acc, c.approx_frobenius_rel)
        for c in rep_b.cells
        if c.kind == "acc"
    }
    # Determinism requires byte-identical core metrics across the two runs.
    assert cores_a == cores_b


def test_pipeline_gate_structure():
    cfg = _small_cfg()
    pipe = EffAttnPipeline(cfg)
    result = pipe.run()
    gate = result["gate"]
    assert set(gate) >= {"sota", "quality_ok", "cost_ok", "grade"}
    assert isinstance(gate["quality_ok"], bool)
    assert isinstance(gate["cost_ok"], bool)
    assert gate["grade"] in {"S", "B", "C"}
    # retrieval non-inferiority ratio is always in [0, 1]
    assert 0.0 <= gate["sota"]["retrieval_ratio"] <= 1.0
    assert gate["sota"]["cost_ratio"] >= 0.0
