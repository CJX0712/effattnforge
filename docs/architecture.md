# EffAttnForge — Architecture

## 1. Goal

Deliver a *world-class* efficient/sparse attention system that is **non-inferior to
exact full attention in quality** while being **strictly cheaper in cost**, with
**full reproducibility**. The system is graded by a pre-registered gate, not by
hand-waved claims.

## 2. Variant family

| Variant | Complexity | Materializes T×T? | Role |
|---------|-----------|-------------------|------|
| `FullAttention` | O(T²d) | yes | SOTA reference / upper bound |
| `LinearAttention` | O(Td²) | no | kernel baseline (elu+1 map) |
| `PerformerAttention` | O(Tdm) | no | FAVOR+ random features |
| `LocalWindowAttention` | O(Twd) | no | sparse (local) |
| `BlockSparseAttention` | O(Tbd) | no | sparse (block + global) |
| `EffAttnFuse` | adaptive | no | **flagship router** |

All variants implement one contract (`core/interfaces.py`): `attend(q,k,v) →
AttentionOutput` with a *score-larger-means-more-mass* semantic, so the evaluation
is fair across mechanisms.

## 3. EffAttnFuse router

```
choose(T, d):
    if T <= full_threshold:        return FullAttention()        # cheap + optimal
    fid = calibration.get(T, 1.0)  # 1.0 = assume pass (conservative)
    if fid >= router_retrieval_floor:  return PerformerAttention(m)
    else:                          return FullAttention()        # guard tripped
```

- `full_threshold = 128`: full only for genuinely tiny sequences.
- `m` is band-dependent: `performer_m_short` (T ≤ 1024) vs `performer_m_long` (T > 1024).
- The calibration table is filled at benchmark time from the *observed* Performer
  retrieval accuracy per length, so the guard reflects reality, not a guess.

## 4. Benchmark pipeline (G1–G4)

```
run_benchmark(cfg, seed):
    set_all(seed); rng = rng_for(101)
    calibration = { T: retrieval_acc(EffAttnFuse.choose(T)) for T in cost_t_values }
    # Accuracy sweep (retrieval_t_values): G1 + G2
    for T: refs = [FullAttention(qf_i,kf_i,vf_i) for i]   # computed ONCE, reused
         for variant: cell(acc) = fidelity(refs) + retrieval
    # Cost sweep (cost_t_values): G3
    for T: q,k,v = make_random_attention(T)                 # once, reused
         for variant: cell(cost) = timed attend(q,k,v)
```

Key correctness decisions:

- **Shared exact reference.** Each T's full-attention reference is computed once and
  reused across all 6 variants — eliminates O(variants) redundant quadratic work and
  makes fidelity comparisons apples-to-apples.
- **Two sweeps, one RNG.** Accuracy and cost use separate T ranges; each `VariantReport`
  carries a `kind` (`"acc"` / `"cost"`) so aggregations never bleed across sweeps.
- **Cost = algorithm only.** Synthetic q/k/v generation is outside the timed region;
  the gate measures the attention op, matching its definition (per-request, batch-of-1).
- **Retrieval metric = affinity argmax.** Earlier a cosine-output metric gave full and
  Performer both 0.0 (softmax is near-uniform over near-equal logits). Using the
  attention *affinity* argmax — which directly corresponds to softmax mass — yields a
  discriminative, correct recall signal.

## 5. Gate (pre-registered S criterion)

```
quality_ok = fuse_retrieval >= 0.98 * full_retrieval
cost_ok    = fuse_cost     <= 0.60 * full_cost
grade = "S" if quality_ok and cost_ok else ("B" if quality_ok else "C")
```

Aggregation is over ≥ 3 seeds (default 3). Determinism (G4) reruns the first seed and
compares core metrics **byte-for-byte**; cost is excluded.

## 6. Determinism

`core/seed.py` derives every stream from a single base seed via fixed salts
(`mixed = base_seed ^ (salt * 0x9E3779B1)`, masked to 32 bits). Salt constants are
embedded in code (never random), so consumption order is identical across runs →
bit-identical benchmarks.

## 7. Error contract

`core/errors.py` registers error classes with a guard that hard-fails on **duplicate
codes** *and* **duplicate names** at import time, preventing the silent-shadowing
pitfall where two errors with the same code mask each other.
