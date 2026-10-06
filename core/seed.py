"""Global deterministic seeding.

Single entry point: ``set_all(seed)`` fixes the global base seed. Every
random stream in the system is derived from it via a *fixed* salt, so that
two runs with the same seed produce byte-identical benchmarks (gate G4).
"""
from __future__ import annotations

import numpy as np

_BASE: dict = {"seed": 42}


def set_all(seed: int) -> int:
    """Set the global base seed and return it."""
    _BASE["seed"] = int(seed)
    return int(seed)


def get_seed() -> int:
    return int(_BASE["seed"])


def rng_for(salt: int) -> np.random.Generator:
    """Deterministic Generator derived from the base seed and a fixed salt.

    Salts are constants in the code (never random), so the consumption order
    is identical across runs.
    """
    mixed = int(_BASE["seed"]) ^ (int(salt) * 0x9E3779B1)
    # Guard against negative seeds confusing Generator.
    mixed &= 0xFFFFFFFF
    return np.random.default_rng(mixed)
