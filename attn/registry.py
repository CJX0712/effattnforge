"""Variant registry: build and enumerate attention implementations."""
from __future__ import annotations

from core.config import Config
from core.errors import InvalidVariantError

from .full import FullAttention
from .fuse import EffAttnFuse
from .linear import LinearAttention
from .performer import PerformerAttention
from .sparse import BlockSparseAttention, LocalWindowAttention


def build_variant(name: str, cfg: Config, calibration: dict[int, float] | None = None):
    if name == "full":
        return FullAttention()
    if name == "linear":
        return LinearAttention()
    if name == "performer":
        return PerformerAttention(m=cfg.performer_m_long, salt=7777)
    if name == "local_window":
        return LocalWindowAttention(window=cfg.window)
    if name == "block_sparse":
        return BlockSparseAttention(block_size=cfg.block_size, n_global=cfg.n_global)
    if name == "effattnfuse":
        return EffAttnFuse(cfg, calibration)
    raise InvalidVariantError(f"unknown variant: {name!r}")


def list_variants(cfg: Config, calibration: dict[int, float] | None = None) -> list[str]:
    """Variant names included in the benchmark (full + efficient + sparse + fuse)."""
    return [
        "full",
        "linear",
        "performer",
        "local_window",
        "block_sparse",
        "effattnfuse",
    ]
