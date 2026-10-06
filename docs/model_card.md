# Model Card — EffAttnFuse (efficient/sparse attention router)

*Prepared in the style of a disciplined model card. EffAttnFuse is a routing
policy, not a trained model; the "training" is deterministic synthetic benchmarking.*

## 1. Model details

- **Name:** EffAttnFuse
- **Type:** length-adaptive attention router (exact full ↔ Performer/FAVOR+)
- **Author:** 晨星 (Chen Xing)
- **License:** MIT
- **Determinism:** fully reproducible (fixed-seed RNG); no learned parameters, no
  pretrained weights, no network access.

## 2. Intended use

- Drop-in *reference* for comparing efficient/sparse attention against exact full
  attention on CPU.
- Research into router design, kernel attention fidelity, and cost/quality trade-offs.
- Educational benchmark demonstrating honest gate evaluation.

## 3. Methodology

- **Synthetic data, seed-reproducible.** Two generators:
  - `make_retrieval` — unit-norm Gaussian keys, in-context associative recall
    (query key equals a random store key; correct output = `value[r]`).
  - `make_random_attention` — unit-norm Q/K (post-LayerNorm-style), random V, for
    the approximation-fidelity (G1) measurement.
- **Gates.**
  - G1 Fidelity: relative Frobenius error of normalized attention vs exact full.
  - G2 Retrieval: in-context associative recall top-1, judged by attention-affinity
    argmax.
  - G3 Cost: aggregate per-request (batch-of-1) wall-clock + analytical peak memory.
  - G4 Determinism: byte-identical core metrics across two same-seed reruns.

## 4. Results (default config, seed 42, n_seeds = 3)

| Variant | Retrieval | Fidelity (rel Frobenius) | Cost (s) | Peak mem (MB) |
|---------|-----------|--------------------------|----------|---------------|
| full | 1.0000 | 0.0000 | 1.69 | 258.0 |
| linear | 0.0208 | 0.0387 | 0.07 | 4.0 |
| performer | 1.0000 | 0.0132 | 0.67 | 32.2 |
| local_window | 0.0382 | 3.4776 | 0.38 | 0.0 |
| block_sparse | 0.2361 | 1.8413 | 0.21 | 0.1 |
| **effattnfuse** | **1.0000** | **0.0176** | **0.62** | **32.2** |

- **SOTA gate:** retrieval ratio 1.0000 (≥ 0.98) ✅; cost ratio **0.3656** (≤ 0.60) ✅.
- **Grade: S.**

## 5. Limitations

- Benchmarked on **synthetic** data; real transformer activations may differ. The
  router is a *reference policy*, not a learned one.
- Performer's wall-clock advantage over exact full only manifests at **long** sequences
  (cross-over ≈ 1024–2048 tokens at d=64 on CPU); at short lengths overhead dominates,
  so the cost sweep intentionally uses large T to make the O(T²) vs O(T) gap honest.
- Single-machine CPU only; no distributed, quantized, or GPU kernels.

## 6. Ethical / environmental considerations

- CPU-only, no GPU/cloud training → negligible energy for the benchmark itself
  (full run ≈ tens of seconds on one core).
- No personal data, no scraped corpora, no model weights shipped.

## 7. Reproducibility

```bash
python -m pip install -r requirements.lock
python examples/run_demo.py   # -> benchmark.json with full gate report
python -m pytest -q          # 32 passing checks incl. determinism
```
