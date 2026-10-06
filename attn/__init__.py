"""Attention variant implementations (effattnforge/attn). Author: 晨星."""
from __future__ import annotations

from .full import FullAttention
from .fuse import EffAttnFuse
from .linear import LinearAttention
from .performer import PerformerAttention
from .registry import build_variant, list_variants
from .sparse import BlockSparseAttention, LocalWindowAttention

__all__ = [
    "BlockSparseAttention",
    "EffAttnFuse",
    "FullAttention",
    "LinearAttention",
    "LocalWindowAttention",
    "PerformerAttention",
    "build_variant",
    "list_variants",
]
