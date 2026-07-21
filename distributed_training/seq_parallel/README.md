# Sequence Parallelism

**Status: code-complete AND locally run — portable, builds on
autograd/matrix.h, col_row_linear's tensor-parallel linear layers, and
networking/collectives.**

## What this measures

PLAN.md Phase 6 step 13: shard the sequence dimension, interleaved with
tensor parallelism (steps 11-12). Two things: LayerNorm's gradient
correctness (isolated), and forward correctness of the full
sequence-parallel <-> tensor-parallel boundary transition.

## Design

LayerNorm normalizes each row (token) independently across the hidden
dimension, so sharding it by sequence position needs **zero
communication** — `layernorm_forward`/`backward` (`layernorm.h`) don't
even know they're being called on a shard rather than a full sequence.
The actual point of this step is the boundary: entering a tensor-parallel
region needs the FULL sequence (`collectives::AllGather`), and leaving one
can produce a sequence SHARD directly via `collectives::ReduceScatter`
instead of an all-reduce followed by a manual re-shard — reusing step 11's
`ColumnParallelLinear`/`RowParallelLinear` for the tensor-parallel region
itself.

## A real bug this caught

Both `AllGather` and `ReduceScatter` place/take a rank's data at a
ROTATED position — rank r's contribution lands at slot `(r+1)%world_size`,
not slot `r` (see `networking/collectives/collectives.h`; ZeRO-2/3 already
had to account for this). Composing the two — an AllGather to enter the
region, a ReduceScatter to leave it — rotates TWICE. The first version of
this test's final reassembly used `(r+1)%world_size` (the position the
*second* rotation, ReduceScatter, gives each rank) to place each rank's
result back into the reference's original row order, and was off by a full
sequence-shard's worth of rows: max abs diff 2.9 against a reference
output whose values were mostly O(1). Working through the composition by
hand (see the comment above the reassembly loop in
`seq_parallel_test.cpp`) shows the two rotations actually CANCEL — rank r
ends up owning the transformed version of its OWN original sequence rows,
`[r*kSeqShard, ...)` — which is also a satisfying invariant: real sequence
parallelism relies on exactly this, a rank keeps working on the same
sequence positions throughout a layer, not some rotated relabeling of them.

## Sanity-run output (Mac, loopback, 2026-07-21)

```
test 1 (LayerNorm gradient check, dx): median relative error = 0.002650, max = 0.019624: PASS
test 2 (sequence-parallel + TP chain, reassembled output): max abs diff = 0.000000: PASS
PASS
```

Test 1 uses the median, not max, for the same reason as
`autograd/autograd_test.cpp` — one element can have an unusually small
finite-difference denominator without indicating a wrong formula; verified
by hand-rederiving `layernorm_backward`'s closed form against the full
chain-rule expansion and confirming they agree exactly.

## Results
TODO: run on GPU hardware — the number that matters is activation memory
saved by sharding LayerNorm/dropout by sequence (the actual point of this
technique — it doesn't reduce compute or communication volume vs. plain
tensor parallelism, it reduces the per-rank activation memory those
non-tensor-parallel ops would otherwise hold in full).

| Sequence length | TP world size | Activation memory (no seq-parallel) | Activation memory (seq-parallel) |
|-------------------|-----------------|----------------------------------------|--------------------------------------|
| TODO | TODO | TODO | TODO |

## Hardware notes
- This step: none required.
- Memory validation (Results table): multi-GPU instance, real sequence
  lengths.
