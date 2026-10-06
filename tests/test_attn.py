"""Tests for every attention variant's contract and a few invariants."""
from __future__ import annotations

import math

import numpy as np
import pytest

from attn.full import FullAttention
from attn.fuse import EffAttnFuse
from attn.linear import LinearAttention, _phi
from attn.performer import PerformerAttention
from attn.registry import build_variant
from attn.sparse import BlockSparseAttention, LocalWindowAttention
from core.config import Config
from core.errors import InvalidVariantError


def _tiny_qkv(T=16, d=8, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((T, d))
    k = rng.standard_normal((T, d))
    v = rng.standard_normal((T, d))
    return q, k, v


def test_full_attention_matches_manual_softmax():
    T, d = 16, 8
    q, k, v = _tiny_qkv(T, d)
    out = FullAttention().attend(q, k, v).output
    # manual reference
    scale = 1.0 / math.sqrt(d)
    scores = (q @ k.T) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    a = e / e.sum(axis=-1, keepdims=True)
    ref = a @ v
    assert np.allclose(out, ref, atol=1e-12)
    # weights are the (T, T) normalized matrix
    w = FullAttention().attend(q, k, v).weights
    assert w.shape == (T, T)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_full_query_affinity_equals_scaled_dot():
    T, d = 16, 8
    q, k, _ = _tiny_qkv(T, d)
    aff = FullAttention().query_affinity(q[-1], k)
    ref = (q[-1] @ k.T) / math.sqrt(d)
    assert np.allclose(aff, ref)
    assert aff.shape == (T,)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: FullAttention(),
        lambda: LinearAttention(),
        lambda: PerformerAttention(m=32, salt=7777),
        lambda: LocalWindowAttention(window=4),
        lambda: BlockSparseAttention(block_size=4, n_global=2),
    ],
)
def test_variant_output_contract(builder):
    T, d = 32, 8
    q, k, v = _tiny_qkv(T, d, seed=3)
    res = builder().attend(q, k, v)
    assert res.output.shape == (T, d)
    assert np.all(np.isfinite(res.output))
    assert res.cost >= 0.0
    assert res.mem_peak_mb >= 0.0
    assert res.variant


def test_linear_attention_feature_map_positive():
    x = np.array([[-2.0, 1.0], [0.5, -3.0]])
    phi = _phi(x)
    assert np.all(phi > 0.0)  # elu(x)+1 is strictly positive


def test_performer_query_affinity_shape_and_finite():
    T, d = 24, 8
    q, k, _ = _tiny_qkv(T, d)
    aff = PerformerAttention(m=32, salt=7777).query_affinity(q[-1], k)
    assert aff.shape == (T,)
    assert np.all(np.isfinite(aff))


def test_sparse_affinity_respects_window_and_blocks():
    T, d = 40, 8
    q, k, _ = _tiny_qkv(T, d, seed=5)
    # Local window: last query (i=T-1) only attends to [T-1-w, T-1]
    w = 4
    laff = LocalWindowAttention(window=w).query_affinity(q[-1], k)
    assert laff.shape == (T,)
    assert np.all(laff[: T - 1 - w] == -np.inf)
    assert np.all(np.isfinite(laff[T - 1 - w :]))
    # Block sparse: only within window+global get finite affinity
    bs = BlockSparseAttention(block_size=8, n_global=4)
    baff = bs.query_affinity(q[-1], k)
    assert baff.shape == (T,)
    assert np.isinf(baff).any()  # some keys are unreachable
    assert np.isfinite(baff).any()


def test_fuse_router_selects_by_threshold():
    cfg = Config(full_threshold=32)
    fuse = EffAttnFuse(cfg, {})
    # T <= threshold -> exact full
    assert fuse.choose(16, 8).name == "full"
    # T > threshold, calibration absent (conservative pass) -> performer
    assert fuse.choose(64, 8).name == "performer"


def test_fuse_router_guard_escalates_on_low_calibration():
    cfg = Config(full_threshold=0)  # force long path
    # calibration below floor -> escalate back to full
    fuse = EffAttnFuse(cfg, {64: 0.5})  # 0.5 < router_retrieval_floor 0.98
    assert fuse.choose(64, 8).name == "full"


def test_fuse_attend_delegates_and_relabels():
    T, d = 64, 8
    q, k, v = _tiny_qkv(T, d, seed=9)
    res = EffAttnFuse(Config(), {}).attend(q, k, v)
    assert res.variant == "effattnfuse"
    assert res.output.shape == (T, d)
    assert np.all(np.isfinite(res.output))


def test_registry_builds_all_variants():
    cfg = Config()
    for name in ["full", "linear", "performer", "local_window", "block_sparse", "effattnfuse"]:
        var = build_variant(name, cfg, {})
        assert var is not None
    with pytest.raises(InvalidVariantError):
        build_variant("does_not_exist", cfg, {})
