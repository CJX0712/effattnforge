"""Configuration with ENV_XXX_* overrides and schema validation.

All knobs are read once at startup from environment variables prefixed
``EFFATTNFORGE_``. Invalid values raise ``ConfigError`` (E100) instead of
silently falling back to a default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from core.errors import ConfigError


@dataclass(frozen=True)
class Config:
    # Determinism
    seed: int = 42
    # Hard safety cap on sequence length (O(T^2) memory guard, pitfalls K).
    max_len: int = 4096
    # Router: use exact full attention at or below this length (cheap + optimal).
    # Above it, Performer(FAVOR+) is non-inferior and far cheaper, so the router
    # switches. 128 keeps full only for genuinely tiny sequences.
    full_threshold: int = 128
    # Performer random-feature dimensions for the two length bands.
    performer_m_short: int = 128
    performer_m_long: int = 256
    # Sparse variants (kept for completeness / router options).
    block_size: int = 64
    window: int = 32
    n_global: int = 8
    # Retrieval benchmark geometry.
    retrieval_d: int = 64
    retrieval_n_samples: int = 24
    retrieval_t_values: tuple[int, ...] = (256, 512, 1024, 2048)
    # Cost sweep uses LARGER T than the accuracy sweep: this is where the O(T^2)
    # (full) vs O(T) (Performer) gap honestly dominates the aggregate cost ratio.
    cost_t_values: tuple[int, ...] = (1024, 2048, 4096)
    # Approximation-fidelity benchmark geometry.
    fidelity_d: int = 32
    fidelity_n_samples: int = 8
    # Quality guard: the router uses Performer only if its calibrated retrieval
    # accuracy (G2) is >= this floor. (Output-level fidelity G1 is reported
    # separately and is intentionally NOT the gate metric.)
    router_retrieval_floor: float = 0.98
    # Number of seeds for the statistical gate (>=3 required; 3 keeps demo <=60s).
    n_seeds: int = 3

    @staticmethod
    def load() -> Config:
        env = os.environ
        kwargs = {}
        int_keys = [
            "seed",
            "max_len",
            "full_threshold",
            "performer_m_short",
            "performer_m_long",
            "block_size",
            "window",
            "n_global",
            "retrieval_d",
            "retrieval_n_samples",
            "fidelity_d",
            "fidelity_n_samples",
            "n_seeds",
        ]
        for k in int_keys:
            raw = env.get(f"EFFATTNFORGE_{k.upper()}")
            if raw is not None:
                try:
                    kwargs[k] = int(raw)
                except ValueError as exc:
                    raise ConfigError(
                        f"EFFATTNFORGE_{k.upper()}={raw!r} is not an int"
                    ) from exc
        raw_t = env.get("EFFATTNFORGE_RETRIEVAL_T_VALUES")
        if raw_t is not None:
            try:
                kwargs["retrieval_t_values"] = tuple(int(x) for x in raw_t.split(",") if x)
            except ValueError as exc:
                raise ConfigError(
                    f"EFFATTNFORGE_RETRIEVAL_T_VALUES={raw_t!r} is not a csv of ints"
                ) from exc
        raw_ct = env.get("EFFATTNFORGE_COST_T_VALUES")
        if raw_ct is not None:
            try:
                kwargs["cost_t_values"] = tuple(int(x) for x in raw_ct.split(",") if x)
            except ValueError as exc:
                raise ConfigError(
                    f"EFFATTNFORGE_COST_T_VALUES={raw_ct!r} is not a csv of ints"
                ) from exc
        raw_floor = env.get("EFFATTNFORGE_PERFORMER_FIDELITY_FLOOR")
        if raw_floor is not None:
            try:
                kwargs["performer_fidelity_floor"] = float(raw_floor)
            except ValueError as exc:
                raise ConfigError(
                    f"EFFATTNFORGE_PERFORMER_FIDELITY_FLOOR={raw_floor!r} not float"
                ) from exc

        cfg = Config(**kwargs)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if not (64 <= self.max_len <= 8192):
            raise ConfigError(f"max_len={self.max_len} must be in [64, 8192]")
        if not (32 <= self.full_threshold <= self.max_len):
            raise ConfigError(
                f"full_threshold={self.full_threshold} must be in [32, max_len]"
            )
        if self.performer_m_short < 16 or self.performer_m_long < self.performer_m_short:
            raise ConfigError("performer_m_short>=16 and <= performer_m_long required")
        if not (1 <= self.window <= 256):
            raise ConfigError(f"window={self.window} must be in [1, 256]")
        if self.retrieval_d < 4:
            raise ConfigError("retrieval_d must be >= 4")
        if self.n_seeds < 3:
            raise ConfigError("n_seeds must be >= 3 (statistical gate needs >=3)")
        if not self.retrieval_t_values:
            raise ConfigError("retrieval_t_values must be non-empty")
        if not self.cost_t_values:
            raise ConfigError("cost_t_values must be non-empty")
        for t in self.retrieval_t_values:
            if t > self.max_len:
                raise ConfigError(f"retrieval T={t} exceeds max_len={self.max_len}")
        for t in self.cost_t_values:
            if t > self.max_len:
                raise ConfigError(f"cost T={t} exceeds max_len={self.max_len}")
