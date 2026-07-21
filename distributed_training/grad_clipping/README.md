# Distributed Gradient Norm Clipping

**Status: code-complete AND locally run — portable, builds on
`networking/ring_allreduce`.**

## What this measures

PLAN.md Phase 6 step 5: distributed gradient norm all-reduce, clip by
global norm.

## Design

Written for the **sharded**-gradient case, not the simpler case where
every rank already holds the complete gradient: `global_grad_norm`
(`grad_clipping.h`) has each rank sum-of-squares its own local shard,
`ring_allreduce`s that one scalar (sum, not concat — cheap: one float per
rank per round regardless of gradient size), then every rank takes
`sqrt()` of the agreed-upon global sum. `clip_grad_by_global_norm` then
scales each rank's local shard by the shared `min(1, max_norm /
global_norm)` factor. Pure data-parallel training doesn't strictly need
the sharded form (the all-reduced gradient is already complete and
identical everywhere), but ZeRO (steps 7-9) shards gradients across ranks
by construction — building this the general way now means those steps
reuse it unchanged instead of needing their own clipping logic later.

## Sanity-run output (Mac, loopback, 2026-07-21)

4000-parameter gradient vector, sharded 1000/rank across 4 simulated
ranks, clipped distributedly and compared to a single-process reference
that never shards:

```
above-threshold (clips): true_norm=323.1583 max_norm=10.0000 scale=0.0309 max_abs_diff=0.000000
below-threshold (no-op): true_norm=0.6282 max_norm=10.0000 scale=1.0000 max_abs_diff=0.000000
PASS
```

Zero difference from the single-process reference in both the clipping and
no-op cases — the sharded computation is exact, not approximate (unlike
step 3/4's ring-order float rounding, this only involves one all-reduced
scalar, so there's no accumulated summation-order drift to speak of).

## Results
TODO: run on GPU hardware — the number that matters is the all-reduce
overhead of the extra one-scalar round-trip per step at real model scale
(thousands of parameter tensors) vs. the cost of the gradient all-reduce
it rides alongside.

| World size | Params | Clip overhead (% of step time) |
|------------|--------|----------------------------------|
| TODO | TODO | TODO |

## Hardware notes
- This step: none required.
- Overhead validation (Results table): multi-GPU instance.
