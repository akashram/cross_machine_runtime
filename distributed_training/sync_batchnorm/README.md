# SyncBatchNorm

**Status: code-complete AND locally run — portable, builds on
autograd/matrix.h and networking/ring_allreduce.**

## What this measures

PLAN.md Phase 6 step 19: SyncBatchNorm with all-reduce of mean/variance.
Validated against a single-process BatchNorm reference over the full
global batch.

## Design

BatchNorm normalizes each FEATURE across the BATCH dimension — the
opposite reduction axis from LayerNorm (step 13, which normalizes each
ROW across features). In data-parallel training, each rank only sees a
LOCAL batch shard; normalizing against local-only statistics is a
different (noisier) computation than a single larger global batch would
produce, especially at small per-rank batch sizes. `sync_batchnorm_forward`
all-reduces local sum/sum-of-squares/count to compute the GLOBAL mean and
variance, then every rank normalizes its own local shard against those
shared statistics. `sync_batchnorm_backward` needs two further all-reduced
quantities beyond the (already-familiar, replicated-parameter) `dgamma`/
`dbeta` sums — `S1`/`S2`, the global sums the closed-form `dx` needs,
since the forward's normalization depended on every sample across every
rank, not just the local ones.

**Reference trick worth calling out**: rather than hand-deriving a second,
independent BatchNorm reference (real risk, per step 15's bug — an
independently-written reference can itself be wrong), the reference here
is `sync_batchnorm_forward`/`backward` called with `world_size=1` over the
full concatenated batch. A 1-rank mesh makes every `ring_allreduce` inside
a no-op (0 rounds — see `ring_allreduce.h`), so this is provably the same
code path reducing to plain BatchNorm, not a second formula that could
independently drift from the real one.

**Communication round-trip count, stated plainly**: forward issues 3
separate all-reduces (sum, sum-of-squares, count), backward issues 4
(`dgamma`, `dbeta`, `S1`, `S2`) — a real implementation would pack these
into fewer wire round-trips (e.g. one all-reduce over a concatenated
buffer). This step validates the distributed statistics math is correct,
not communication-count optimization, which is a real but separate
follow-up.

## Sanity-run output (Mac, loopback, 2026-07-21)

4 ranks, 5 samples/rank (20 total), 6 features, vs. the world_size=1
reference over the same 20 samples:

```
  forward    max abs diff = 0.000000: PASS
  dx         max abs diff = 0.000001: PASS
  dgamma     max abs diff = 0.000006: PASS
  dbeta      max abs diff = 0.000002: PASS
PASS
```

Matches to float32 precision (the ~1e-6 differences are ring-summation-
order noise, same as steps 3/7-9's training-loop comparisons, not a
correctness gap).

## Results
TODO: run on GPU hardware — the number that matters is all-reduce
overhead (7 round-trips per step, unpacked) as a fraction of step time at
real batch/feature scale, and whether packing them into fewer round-trips
matters in practice at that scale.

| Batch/rank | Features | World size | Unpacked overhead | Packed overhead (if implemented) |
|--------------|------------|--------------|------------------------|----------------------------------------|
| TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required.
- Overhead validation (Results table): multi-GPU instance.
