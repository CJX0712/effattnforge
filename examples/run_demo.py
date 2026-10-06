"""End-to-end demo: run the benchmark, evaluate the S gate, verify determinism.

Usage:
    python examples/run_demo.py [--seed 42] [--out benchmark.json]

Determinism (gate G4): the benchmark is rerun for the first seed and the
core metrics (retrieval_acc, approx_frobenius_rel) are compared byte-for-byte.
Wall-clock cost is intentionally EXCLUDED from the determinism assertion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime

# Ensure the repo root (parent of examples/) is importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.config import Config
from core.seed import set_all
from core.types import BenchmarkReport
from eval.benchmark import run_benchmark
from pipeline import EffAttnPipeline

VARIANT_ORDER = ["full", "linear", "performer", "local_window", "block_sparse", "effattnfuse"]


def _agg_per_variant(reports: list[BenchmarkReport]) -> dict[str, dict[str, float]]:
    out = {}
    for name in VARIANT_ORDER:
        acc_cells = [
            c for r in reports for c in r.cells if c.variant == name and c.kind == "acc"
        ]
        cost_cells = [
            c for r in reports for c in r.cells if c.variant == name and c.kind == "cost"
        ]
        if not acc_cells:
            continue
        out[name] = {
            "retrieval_acc": float(np.mean([c.retrieval_acc for c in acc_cells])),
            "fidelity_rel": float(np.mean([c.approx_frobenius_rel for c in acc_cells])),
            "cost": (
                float(np.sum([c.cost for c in cost_cells])) if cost_cells else float("nan")
            ),
            "mem_peak_mb": (
                float(np.max([c.mem_peak_mb for c in cost_cells])) if cost_cells else 0.0
            ),
        }
    return out


def _per_t_table(reports: list[BenchmarkReport]) -> list[dict]:
    rows = []
    for name in VARIANT_ORDER:
        cells = [c for r in reports for c in r.cells if c.variant == name]
        by_t = {}
        for c in cells:
            by_t.setdefault(c.T, []).append(c)
        for T in sorted(by_t):
            grp = by_t[T]
            acc = [x for x in grp if x.kind == "acc"]
            cost = [x for x in grp if x.kind == "cost"]
            rows.append(
                {
                    "variant": name,
                    "T": T,
                    "retrieval_acc": (
                        float(np.mean([x.retrieval_acc for x in acc])) if acc else None
                    ),
                    "fidelity_rel": (
                        float(np.mean([x.approx_frobenius_rel for x in acc])) if acc else None
                    ),
                    "cost": float(np.sum([x.cost for x in cost])) if cost else None,
                    "mem_peak_mb": float(np.max([x.mem_peak_mb for x in cost])) if cost else None,
                }
            )
    return rows


def _determinism_ok(report_a: BenchmarkReport, report_b: BenchmarkReport) -> dict:
    cores_a = {
        (c.variant, c.T, c.kind): (c.retrieval_acc, c.approx_frobenius_rel)
        for c in report_a.cells
    }
    cores_b = {
        (c.variant, c.T, c.kind): (c.retrieval_acc, c.approx_frobenius_rel)
        for c in report_b.cells
    }
    max_abs = 0.0
    keys = set(cores_a) | set(cores_b)
    for k in keys:
        if k not in cores_a or k not in cores_b:
            return {"bit_identical": False, "max_abs_delta_core": float("nan")}
        for va, vb in zip(cores_a[k], cores_b[k], strict=True):
            max_abs = max(max_abs, abs(va - vb))
    return {"bit_identical": max_abs == 0.0, "max_abs_delta_core": max_abs}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="EffAttnForge demo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="benchmark.json")
    ap.add_argument("--n-seeds", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.load()
    if args.n_seeds is not None:
        cfg = Config(**{**cfg.__dict__, "n_seeds": args.n_seeds})

    set_all(args.seed)
    pipe = EffAttnPipeline(cfg)
    result = pipe.run()
    reports: list[BenchmarkReport] = result["reports"]
    gate = result["gate"]

    # Determinism: rerun first seed, compare core metrics (cost excluded).
    dup = run_benchmark(cfg, result["seeds"][0])
    det = _determinism_ok(reports[0], dup)

    per_var = _agg_per_variant(reports)
    per_t = _per_t_table(reports)

    print("=" * 78)
    print(" EffAttnForge — Efficient / Sparse Attention Benchmark")
    print("=" * 78)
    print(f" seed={args.seed}  n_seeds={cfg.n_seeds}")
    print(f" accuracy T={cfg.retrieval_t_values}   cost T={cfg.cost_t_values}")
    print("-" * 78)
    hdr = f"{'variant':<14}{'retrieval':>11}{'fidelity':>11}{'cost(s)':>11}{'memMB':>9}"
    print(hdr)
    for name in VARIANT_ORDER:
        if name not in per_var:
            continue
        s = per_var[name]
        print(
            f"{name:<14}{s['retrieval_acc']:>11.4f}{s['fidelity_rel']:>11.4f}"
            f"{s['cost']:>11.2f}{s['mem_peak_mb']:>9.1f}"
        )
    print("-" * 78)
    sota = gate["sota"]
    fa = sota["fuse_retrieval_acc"]
    fb = sota["full_retrieval_acc"]
    rr = sota["retrieval_ratio"]
    fc = sota["fuse_aggregate_cost"]
    fcc = sota["full_aggregate_cost"]
    cr = sota["cost_ratio"]
    print(" SOTA gate (vs exact full attention):")
    print(
        f"   retrieval  fuse={fa:.4f}  full={fb:.4f}  ratio={rr:.4f}"
        f"  (need >=0.98) -> {'PASS' if gate['quality_ok'] else 'FAIL'}"
    )
    print(
        f"   cost       fuse={fc:.2f}s  full={fcc:.2f}s  ratio={cr:.4f}"
        f"  (need <=0.60) -> {'PASS' if gate['cost_ok'] else 'FAIL'}"
    )
    print(
        f"   determinism: bit_identical={det['bit_identical']}"
        f"  max|Δcore|={det['max_abs_delta_core']:.2e}"
    )
    print(f" GRADE = {gate['grade']}")
    print("=" * 78)

    payload = {
        "system": "EffAttnForge",
        "author": "晨星",
        "seed": args.seed,
        "config": {
            k: list(v) if isinstance(v, tuple) else v for k, v in cfg.__dict__.items()
        },
        "per_variant": per_var,
        "per_T": per_t,
        "gate": gate,
        "determinism": det,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f" wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
