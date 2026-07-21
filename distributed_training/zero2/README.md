# ZeRO Stage 2

**Status: code-complete AND locally run — portable, builds on autograd/,
zero1/adam.h, and networking/collectives.**

## What this measures

PLAN.md Phase 6 step 8: shard gradients in addition to optimizer state
(step 7 sharded only optimizer state, keeping gradients fully replicated
via all-reduce). Correctness against the same no-ZeRO baseline step 7 used.

## Design

`ZeroStage2Optimizer::step()` (`zero2_optimizer.h`) takes each rank's own
LOCAL, un-reduced gradient and `collectives::ReduceScatter`s it in place —
every rank ends up owning exactly one chunk's true sum (the rest of the
buffer is left as garbage, which is why the parameter is a non-const
reference: a rank never needs to materialize the full reduced gradient,
that's the whole memory-saving point). That owned chunk is Adam-updated,
then `AllGather`-ed back to a full parameter vector, same as ZeRO-1.

**Shard-ownership convention deliberately differs from ZeRO-1**: ZeRO-1
assigns "rank r owns shard r" and has to un-rotate `AllGather`'s native
"rank r's contribution lands at slot `(r+1)%world_size`" placement to
keep that assignment consistent. Here, "rank r owns shard
`(r+1)%world_size`" is used instead — exactly the chunk `ReduceScatter`
already gives rank r ownership of, and exactly the slot `AllGather` places
it at, so *no* remapping is needed in either direction. Both conventions
are equally correct (the assignment of which rank owns which shard is
arbitrary either way); this one is simpler specifically because gradient
sharding, unlike ZeRO-1's optimizer-only sharding, has a natural ownership
assignment falling out of `ReduceScatter` itself.

## Sanity-run output (Mac, loopback, 2026-07-21)

Identical setup to step 7 (same toy MLP, same dataset, same seeds) so the
two are directly comparable:

```
epoch  baseline_loss  zero2_loss  rel_diff
    0       1.879450    1.879450    0.0000
   10       0.217109    0.217109    0.0000
   20       0.054142    0.054142    0.0000
   30       0.008090    0.008090    0.0000
   40       0.001958    0.001958    0.0000
   50       0.000870    0.000870    0.0000
final: baseline=0.000581 zero2=0.000581
PASS
```

Numbers match step 7's ZeRO-1 run exactly — expected, since both compute
the same mathematical gradient and apply the same Adam update; only which
rank materializes which piece of state differs.

## Results
TODO: run on GPU hardware — the number that matters is gradient memory per
rank on top of step 7's optimizer-state savings (ZeRO-2 should cut both to
`1/world_size` of the no-ZeRO baseline) at real model scale.

| World size | Grad+optimizer memory / rank (no ZeRO) | Grad+optimizer memory / rank (ZeRO-2) |
|------------|------------------------------------------|------------------------------------------|
| TODO | TODO | TODO |

## Hardware notes
- This step: none required.
- Memory validation (Results table): GPU instance, real model scale.
