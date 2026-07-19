# Memory Planning Pass

**Status: code-complete, not yet built — requires MLIR on Linux.**

## What this measures
Liveness analysis on tensor values to identify non-overlapping lifetimes,
then aliases buffers to minimize peak HBM usage. Compare vs XLA memory planning.

## Design
Interval-based buffer assignment (`MemPlanningPass.cpp`), the same shape as
XLA's BufferAssignment: program order is the timeline, each statically-shaped
tensor value gets a `[defTime, lastUseTime]` interval, and a first-fit sweep
over a coalescing free list assigns shared-arena offsets so overlapping
intervals never alias. The pass is annotation-only — it sets
`runtime.buffer_offset` on each producing op and `runtime.peak_memory_bytes`
on the function, and leaves realizing those offsets (arena mmap vs.
`memref.alloc` vs. target-specific allocator) to the AOT pipeline (step 12).
Dynamically-shaped values are skipped (not planned) since their size isn't
known until runtime. Runs *before* rematerialization (step 8) in the
pipeline — remat can shrink an interval's live range, which is why placement
re-runs are cheap: the arena sweep, not the whole pass, is what a remat
decision invalidates.

## Results
TODO: run on Linux with MLIR.

| Model | Naive peak mem (GB) | Planned peak mem (GB) | Reduction |
|-------|---------------------|-----------------------|-----------|
| MLP 4-layer | TODO | TODO | TODO |
| Transformer block | TODO | TODO | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
