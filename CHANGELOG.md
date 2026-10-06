# Changelog

All notable changes to EffAttnForge are documented here. The project follows a
grade-gated release policy: a release is cut only when the pre-registered S gate
(quality non-inferiority **and** strictly cheaper cost, with full determinism) is met.

## [0.1.0] — 2026-10-06

Initial S-grade release.

### Added
- Six attention variants: `FullAttention` (reference), `LinearAttention`,
  `PerformerAttention` (FAVOR+), `LocalWindowAttention`, `BlockSparseAttention`, and
  the flagship router `EffAttnFuse`.
- Four-gate benchmark (G1 fidelity, G2 retrieval, G3 cost, G4 determinism) over ≥ 3
  seeds.
- Length-adaptive router with a calibration-backed quality guard.
- Pre-registered S gate: retrieval ≥ 0.98× full **and** aggregate wall-clock
  ≤ 0.60× full. **Achieved: Grade S** (retrieval 1.0000, cost ratio 0.3656,
  bit-identical determinism).
- Deterministic RNG core (`core/seed.py`) with fixed salts.
- Error registry with duplicate-code / duplicate-name guards (`core/errors.py`).
- Unit-test suite (32 tests) covering core, attn, eval, determinism, and pipeline.
- CI workflow (ruff + pytest + smoke benchmark), `Makefile`, `requirements.lock`,
  `Dockerfile` hooks, and `gh_push.py` (three-tier degrade push).

### Methodology notes
- Exact full-attention reference computed once per (T, sample) and reused across
  variants (apples-to-apples, ~6× less quadratic work).
- Two decoupled sweeps (accuracy vs cost) sharing one RNG; cells tagged by `kind`.
- Cost timed on the attention op only; synthetic data generation kept outside the
  timed region.
- Performer random-feature variance scaled by `d^{-1/4}` to match softmax temperature
  `1/√d`, yielding faithful kernel approximation (G1 ≈ 0.013).
