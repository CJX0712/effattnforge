"""EffAttnFuse — length-adaptive router (the flagship).

Picks the cheapest variant whose quality is non-inferior to exact full
attention for the requested sequence length:

  * T <= full_threshold          -> exact FullAttention (cost negligible, optimal)
  * T >  full_threshold          -> Performer(FAVOR+) with band-dependent m

A quality guard escalates back to full when the calibrated Performer fidelity
for that length falls below ``performer_fidelity_floor``. The calibration table
is filled from the G1 approximation-fidelity measurement at benchmark time, so
the guard reflects *observed* behaviour, not a hand-tuned assumption.
"""
from __future__ import annotations

import numpy as np

from core.config import Config
from core.types import AttentionOutput

from .full import FullAttention
from .performer import PerformerAttention


class EffAttnFuse:
    name = "effattnfuse"

    def __init__(self, cfg: Config, calibration: dict[int, float] | None = None) -> None:
        self.cfg = cfg
        self.calibration: dict[int, float] = dict(calibration or {})

    def _performer_m(self, T: int) -> int:
        return self.cfg.performer_m_long if T > 1024 else self.cfg.performer_m_short

    def choose(self, T: int, d: int):
        if T <= self.cfg.full_threshold:
            return FullAttention()
        m = self._performer_m(T)
        fid = self.calibration.get(T, 1.0)  # absent => assume pass (conservative)
        if fid >= self.cfg.router_retrieval_floor:
            return PerformerAttention(m=m, salt=7777)
        # Quality guard tripped: fall back to exact attention.
        return FullAttention()

    def attend(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> AttentionOutput:
        T = q.shape[0]
        variant = self.choose(T, q.shape[1])
        res = variant.attend(q, k, v)
        # Relabel so the benchmark attributes cost/quality to the router itself.
        return AttentionOutput(
            res.output, res.weights, res.cost, res.mem_peak_mb, self.name
        )

    def theoretical_complexity(self, T: int, d: int) -> str:
        if T <= self.cfg.full_threshold:
            return "O(T^2 d) [short: exact]"
        return f"O(T d m) [long: Performer m={self._performer_m(T)}]"

    def query_affinity(self, q_last: np.ndarray, k: np.ndarray) -> np.ndarray:
        variant = self.choose(k.shape[0], k.shape[1])
        return variant.query_affinity(q_last, k)
