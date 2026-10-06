"""Benchmark pipeline: G1 fidelity, G2 retrieval, G3 cost, G4 determinism.

Gate metrics (per seed, then aggregated over >=3 seeds):
  G1  approx_frobenius_rel : variant output vs exact full attention output
  G2  retrieval_acc        : in-context associative-recall top-1 over samples
  G3  cost / mem_peak_mb   : per-request (batch-of-1) wall-clock + analytical peak mem
  G4  determinism          : byte-identical core metrics across two same-seed runs
                           (cost is EXCLUDED from the determinism comparison)

Two decoupled sweeps share a single deterministic RNG stream:
  * accuracy sweep (retrieval_t_values): G1 fidelity + G2 retrieval. The exact
    full-attention *reference* is computed ONCE per (T, sample) and reused across
    all variants (apples-to-apples, and ~6x less quadratic work).
  * cost sweep (cost_t_values): G3 wall-clock. Uses larger T so the O(T^2) vs
    O(T) gap dominates the aggregate cost ratio honestly.

All randomness flows from one base seed via fixed salts, so reruns are
reproducible. No internet, no pretrained weights.
"""
from __future__ import annotations

import time

import numpy as np

from attn.full import FullAttention
from attn.registry import build_variant
from core.config import Config
from core.seed import rng_for, set_all
from core.types import BenchmarkReport, VariantReport
from data import make_random_attention, make_retrieval
from eval import rel_frobenius, retrieval_hit_from_affinity


def _retrieval_of(variant, T: int, cfg: Config, rng) -> float:
    """Retrieval top-1 accuracy of a variant at length T (calibrates the router)."""
    ks, _, r = make_retrieval(rng, T, cfg.retrieval_d, cfg.retrieval_n_samples)
    hits = 0
    for s in range(ks.shape[0]):
        aff = variant.query_affinity(ks[s][T - 1], ks[s])
        hits += retrieval_hit_from_affinity(aff, int(r[s]), T)
    return hits / ks.shape[0]


def _accuracy_cell(variant, T, cfg, qf, kf, vf, refs, ks, r) -> VariantReport:
    # G1 fidelity: variant output vs the exact full-attention reference (precomputed).
    errs = []
    for i in range(qf.shape[0]):
        out = variant.attend(qf[i], kf[i], vf[i]).output
        errs.append(rel_frobenius(out, refs[i]))
    fid = float(np.mean(errs))
    # G2 retrieval: in-context associative recall via attention-affinity argmax.
    hits = 0
    for s in range(ks.shape[0]):
        aff = variant.query_affinity(ks[s][T - 1], ks[s])  # (T,)
        hits += retrieval_hit_from_affinity(aff, int(r[s]), T)
    acc = hits / ks.shape[0]
    return VariantReport(
        variant=variant.name,
        T=T,
        approx_frobenius_rel=fid,
        retrieval_acc=acc,
        cost=0.0,
        mem_peak_mb=0.0,
        kind="acc",
    )


def _cost_cell(variant, T, q, k, v) -> VariantReport:
    # Timed region measures ONLY the attention forward pass. The q/k/v tensors
    # are produced by the upstream model in real serving; synthesizing them is
    # benchmark scaffolding generated outside this window, so the gate reflects
    # the algorithm's own cost (full = O(T^2) vs Performer = O(T)).
    t0 = time.perf_counter()
    res = variant.attend(q, k, v)
    cost = time.perf_counter() - t0
    return VariantReport(
        variant=variant.name,
        T=T,
        approx_frobenius_rel=0.0,
        retrieval_acc=0.0,
        cost=cost,
        mem_peak_mb=res.mem_peak_mb,
        kind="cost",
    )


def run_benchmark(cfg: Config, seed: int) -> BenchmarkReport:
    set_all(seed)
    rng = rng_for(salt=101)
    # Build calibration: measure the retrieval accuracy the router's chosen
    # variant would achieve (the guard escalates to full if it falls short).
    calibration: dict = {}
    for T in cfg.cost_t_values:
        probe = build_variant("effattnfuse", cfg, {})
        chosen = probe.choose(T, cfg.retrieval_d)
        if chosen.name == "performer":

            calibration[T] = _retrieval_of(chosen, T, cfg, rng)
        else:
            calibration[T] = 1.0  # exact full always passes the guard
    roster = ["full", "linear", "performer", "local_window", "block_sparse", "effattnfuse"]
    cells: list = []

    # ---- Accuracy sweep: G1 + G2 (shared exact reference per T) -------------
    for T in cfg.retrieval_t_values:
        qf, kf, vf = make_random_attention(rng, T, cfg.fidelity_d, cfg.fidelity_n_samples)
        full_ref = FullAttention()
        refs = [full_ref.attend(qf[i], kf[i], vf[i]).output for i in range(qf.shape[0])]
        ks, _, r = make_retrieval(rng, T, cfg.retrieval_d, cfg.retrieval_n_samples)
        for name in roster:
            var = build_variant(name, cfg, calibration)
            cells.append(_accuracy_cell(var, T, cfg, qf, kf, vf, refs, ks, r))

    # ---- Cost sweep: G3 (larger T so O(T^2) vs O(T) dominates) --------------
    for T in cfg.cost_t_values:
        # Cost data generated once per T and reused across variants (same input
        # => fair comparison); it is NOT part of the timed region.
        qc, kc, vc = make_random_attention(rng, T, cfg.retrieval_d, 1)
        for name in roster:
            var = build_variant(name, cfg, calibration)
            cells.append(_cost_cell(var, T, qc[0], kc[0], vc[0]))

    return BenchmarkReport(seed=seed, cells=cells)


# ---- Aggregate gate helpers ------------------------------------------------
#
# Each metric reads only its own cell kind, so the two sweeps never bleed into
# one another (accuracy cells carry cost=0, cost cells carry fidelity/retrieval
# placeholders).

CORE_FIELDS = ("approx_frobenius_rel", "retrieval_acc")


def aggregate_retrieval(reports: list, variant: str) -> float:
    vals = [
        c.retrieval_acc
        for rep in reports
        for c in rep.cells
        if c.variant == variant and c.kind == "acc"
    ]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate_cost(reports: list, variant: str) -> float:
    vals = [
        c.cost
        for rep in reports
        for c in rep.cells
        if c.variant == variant and c.kind == "cost"
    ]
    return float(np.sum(vals)) if vals else float("nan")


def aggregate_fidelity(reports: list, variant: str) -> float:
    vals = [
        c.approx_frobenius_rel
        for rep in reports
        for c in rep.cells
        if c.variant == variant and c.kind == "acc"
    ]
    return float(np.mean(vals)) if vals else float("nan")
