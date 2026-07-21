# ZeRO Stage 3

**Status: code-complete AND locally run — portable, builds on autograd/,
zero1/adam.h, and networking/collectives.**

## What this measures

PLAN.md Phase 6 step 9: shard parameters, on top of step 8's gradient and
optimizer-state sharding. Correctness against the same no-ZeRO baseline
steps 7-8 used, plus the property that actually defines ZeRO-3: a rank's
persistent, between-step memory holds only its `1/world_size` shard of
everything (params, grads, optimizer state) — never a full copy.

## Design

`ZeroStage3Optimizer` (`zero3_optimizer.h`) holds exactly one persistent
field beyond its Adam state: `params_shard_`, this rank's owned slice. It
exposes two operations: `gather_full_params()` (`AllGather`s the shard
into a full vector, transient — used to build a local model just for this
step's forward/backward, then discarded) and `step()` (reduce-scatters the
local gradient, Adam-updates `params_shard_` in place, same as ZeRO-2 —
notably does NOT reassemble or store a full parameter vector; next step
calls `gather_full_params()` again for a fresh one).

**Simplification vs. a real ZeRO-3, stated plainly**: a real
implementation gathers and releases parameters PER LAYER (just before that
layer's forward, again just before its backward), so PEAK memory during a
single step also stays near `1/world_size`. This gathers the WHOLE
parameter vector once per step instead — simpler, and correctness-
equivalent (the training math does not care when within a step the gather
happens), but it validates ZeRO-3's between-step memory floor and
correctness, not its per-layer peak-memory behavior. That per-layer
benefit only matters under real memory pressure at real model scale — GPU
territory, see Results below.

## Sanity-run output (Mac, loopback, 2026-07-21)

Identical setup to steps 7-8:

```
epoch  baseline_loss  zero3_loss  rel_diff
    0       1.879450    1.879450    0.0000
   10       0.217109    0.217109    0.0000
   20       0.054142    0.054142    0.0000
   30       0.008090    0.008090    0.0000
   40       0.001958    0.001958    0.0000
   50       0.000870    0.000870    0.0000
final: baseline=0.000581 zero3=0.000581
PASS
```

Identical numbers to steps 7 and 8 — expected, all three compute the same
gradient and apply the same Adam update; only how much state each rank
persists between steps differs.

## Results
TODO: run on GPU hardware — the numbers that matter are (1) per-layer
peak memory during forward/backward with just-in-time gather/release (the
simplification this Mac run does not exercise) and (2) confirmation that
ZeRO-3 enables training a model too large to fit on a single GPU at all
(PLAN.md's stated definition of done for this phase).

| World size | Persistent memory / rank (no ZeRO) | Persistent memory / rank (ZeRO-3) | Max trainable model size |
|------------|--------------------------------------|---------------------------------------|-----------------------------|
| TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required for the correctness/between-step-memory
  property validated here.
- Per-layer peak-memory and max-model-size validation (Results table):
  GPU instance, real model scale.
