"""Pipeline orchestration for EffAttnForge."""
from __future__ import annotations

from core.config import Config
from core.types import BenchmarkReport
from eval.benchmark import (
    aggregate_cost,
    aggregate_fidelity,
    aggregate_retrieval,
    run_benchmark,
)


class EffAttnPipeline:
    """Run the multi-seed benchmark and evaluate the S-grade gate."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, seeds: list[int] | None = None) -> dict:
        if seeds is None:
            seeds = [self.cfg.seed + i for i in range(self.cfg.n_seeds)]
        reports: list[BenchmarkReport] = [run_benchmark(self.cfg, s) for s in seeds]
        gate = self._evaluate_gate(reports)
        return {"reports": reports, "gate": gate, "seeds": seeds, "cfg": self.cfg}

    def _evaluate_gate(self, reports: list[BenchmarkReport]) -> dict:
        fuse_acc = aggregate_retrieval(reports, "effattnfuse")
        full_acc = aggregate_retrieval(reports, "full")
        fuse_cost = aggregate_cost(reports, "effattnfuse")
        full_cost = aggregate_cost(reports, "full")
        fuse_fid = aggregate_fidelity(reports, "effattnfuse")
        full_fid = aggregate_fidelity(reports, "full")  # 0 by construction

        quality_ok = bool(fuse_acc >= 0.98 * full_acc)
        cost_ok = bool(fuse_cost <= 0.60 * full_cost)
        sota = {
            "fuse_retrieval_acc": fuse_acc,
            "full_retrieval_acc": full_acc,
            "retrieval_ratio": (fuse_acc / full_acc) if full_acc else float("nan"),
            "fuse_aggregate_cost": fuse_cost,
            "full_aggregate_cost": full_cost,
            "cost_ratio": (fuse_cost / full_cost) if full_cost else float("nan"),
            "fuse_fidelity_rel": fuse_fid,
            "full_fidelity_rel": full_fid,
        }
        # Pre-registered S gate: non-inferior quality AND strictly cheaper cost.
        if quality_ok and cost_ok:
            grade = "S"
        elif quality_ok:
            grade = "B"
        else:
            grade = "C"
        return {
            "sota": sota,
            "quality_ok": quality_ok,
            "cost_ok": cost_ok,
            "grade": grade,
        }
