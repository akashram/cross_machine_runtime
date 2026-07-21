# ZeRO Stage 1

**Status: code-complete AND locally run — portable, builds on autograd/,
networking/ring_allreduce, and networking/collectives.**

## What this measures

PLAN.md Phase 6 step 7: shard optimizer state (Adam momentum + variance)
across data-parallel ranks. Memory reduction and correctness — the loss
curve must match a no-ZeRO baseline exactly (ZeRO changes WHERE optimizer
state lives, not the training math).

## Design

`ZeroStage1Optimizer` (`zero1_optimizer.h`): each rank owns Adam state
(`adam.h`) for only its `1/world_size` shard of the parameter vector.
Every step: (1) each rank computes a local gradient from its own data
shard (as in step 3) and `ring_allreduce`s it so every rank has the
complete, identical gradient; (2) `ZeroStage1Optimizer::step()` updates
only this rank's own parameter shard using only its own Adam state, then
`collectives::AllGather`s the updated shards so every rank ends the step
with the full, identical, updated parameter vector for the next forward
pass.

Two things worth flagging:
- **AllGather's rotation**: `collectives::AllGather` places rank r's
  contribution at slot `(r+1) % world_size` in the output (inherited from
  the ring all-gather it's built on — see `networking/collectives/collectives.h`).
  `step()` un-rotates when writing back into `full_params` so shard *i*
  always corresponds to parameters `[i*shard_size, (i+1)*shard_size)`
  regardless of which rank is reading it.
- **Padding for uniform shards**: `AllGather` requires the same
  `send_count` from every rank, so the parameter count is padded up to a
  multiple of `world_size`; `mlp.h`'s `unflatten_params` only reads the
  real prefix, so the padding tail is inert.

`mlp.h` grew `flatten_params`/`flatten_grads`/`unflatten_params` for this
step — the glue between `autograd::Tensor` parameter lists and the flat
`float*` buffers `ring_allreduce`/`collectives` operate on. Every later
step needing to shard or communicate model state reuses these.

## Sanity-run output (Mac, loopback, 2026-07-21)

Toy MLP (2 -> 8 -> 3), 3-class Gaussian-blob classification, 120 samples
(30/rank), 60 epochs, Adam lr=0.05. Baseline: single-process Adam, full
optimizer state. ZeRO-1: 4 simulated ranks (real TCP loopback threads,
independent memory per rank — no shared state to cheat with), sharded
Adam state.

```
epoch  baseline_loss  zero1_loss  rel_diff
    0       1.879450    1.879450    0.0000
   10       0.217109    0.217109    0.0000
   20       0.054142    0.054142    0.0000
   30       0.008090    0.008090    0.0000
   40       0.001958    0.001958    0.0000
   50       0.000870    0.000870    0.0000
final: baseline=0.000581 zero1=0.000581
PASS
```

Loss curves match to 6 decimal places at every checkpoint — tighter than
step 3-5's SGD comparisons even, because with equal-sized shards averaging
four per-shard Adam-input means reduces to the same float32 arithmetic
either way (see the scaling note in `zero1_test.cpp`).

## Results
TODO: run on GPU hardware — the number that matters is peak memory per
rank (ZeRO-1 should cut optimizer-state memory to `1/world_size` of the
no-ZeRO baseline: 2x parameter memory, not 2x*world_size) at real model
scale.

| World size | Optimizer state / rank (no ZeRO) | Optimizer state / rank (ZeRO-1) |
|------------|-----------------------------------|-----------------------------------|
| TODO | TODO | TODO |

## Hardware notes
- This step: none required.
- Memory validation (Results table): GPU instance, real model scale.
