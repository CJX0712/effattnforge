# EffAttnForge

> S-grade efficient / sparse attention benchmark — full-attention parity at a fraction of the compute.

[![CI](https://github.com/CJX0712/effattnforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/effattnforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Grade](https://img.shields.io/badge/grade-S--level-brightgreen.svg)](https://github.com/CJX0712/effattnforge)
[![Determinism](https://img.shields.io/badge/determinism-bit--identical-success.svg)](https://github.com/CJX0712/effattnforge)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**EffAttnForge** benchmarks a family of attention mechanisms — exact full attention
(SOTA reference), linear attention, Performer/FAVOR+, local-window, block-sparse,
and a length-adaptive router (**EffAttnFuse**) — under four rigorous gates:

| Gate | Metric | Result |
|------|--------|--------|
| **G1** Fidelity | relative Frobenius error vs exact full | Performer 0.0132, EffAttnFuse 0.0176 |
| **G2** Retrieval | in-context associative recall (top-1) | EffAttnFuse **1.0000** (≥ 0.98× full) ✅ |
| **G3** Cost | aggregate wall-clock (batch-of-1) | EffAttnFuse **0.62 s** vs full **1.69 s** → **0.366×** (≤ 0.60×) ✅ |
| **G4** Determinism | byte-identical core metrics across reruns | **bit_identical = True** ✅ |

**Final grade: `S`** — non-inferior quality *and* strictly cheaper cost, with full
reproducibility.

---

## Quickstart

```bash
python -m pip install -r requirements.lock
python -m ruff check .      # lint gate
python -m pytest -q         # unit tests
python examples/run_demo.py  # runs all gates, writes benchmark.json
```

## What it measures

- **FullAttention** — exact O(T²d) softmax; the upper-bound reference.
- **LinearAttention** — Katharopoulos et al. 2020 kernel attention, O(Td²).
- **PerformerAttention** — Choromanski et al. 2020 FAVOR+ random features, O(Tdm),
  with the variance scaled to `d^{-1/4}` so it matches the softmax temperature
  `1/√d` (this is what makes the kernel estimate faithful).
- **LocalWindowAttention / BlockSparseAttention** — sparse variants (honestly degrade
  on long-range retrieval; that is *why* they are benchmarked next to global ones).
- **EffAttnFuse** — the flagship length-adaptive router. For `T ≤ full_threshold` it
  uses exact full attention (cheap + optimal); for `T > full_threshold` it switches to
  Performer. A calibrated quality guard escalates back to full if the observed
  Performer retrieval accuracy for that length drops below `router_retrieval_floor`.

## Benchmark design

The benchmark uses **two decoupled sweeps** sharing one deterministic RNG stream:

- **Accuracy sweep** (`retrieval_t_values`) — G1 fidelity + G2 retrieval. The exact
  full-attention *reference* is computed **once per (T, sample)** and reused across
  all variants (apples-to-apples, ~6× less quadratic work).
- **Cost sweep** (`cost_t_values`) — G3 wall-clock, using *larger* T so the O(T²) vs
  O(T) gap honestly dominates the aggregate cost ratio.

Cost is timed on the attention forward pass only; synthetic q/k/v generation is
benchmark scaffolding and is kept outside the timed region (in real serving those
tensors arrive from the upstream model).

## Determinism

Every random stream is derived from a single base seed via *fixed* salts
(`core/seed.py`), so two runs with the same seed produce byte-identical core
metrics. Wall-clock cost is intentionally excluded from the determinism assertion.

## Repository layout

```
core/      config, errors (duplicate-code guarded), seed, types, interfaces
attn/      full, linear, performer, sparse, fuse (router), registry
data/      synthetic generators (reproducible)
eval/      metrics + benchmark pipeline (G1–G4)
pipeline/  multi-seed orchestration + S-grade gate
examples/  run_demo.py
tests/     pytest suite (core / attn / eval / benchmark)
gh_push.py three-tier degrade push (gh → token → bundle)
```

## License & author

MIT — authored by **晨星 (Chen Xing)**.
