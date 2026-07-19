# Rematerialization Pass (Activation Checkpointing)

**Status: code-complete, not yet built — requires MLIR on Linux.**

## What this measures
Identifies tensors cheaper to recompute than store (activations with low compute/memory ratio).
Inserts recompute ops at backward pass entry points to trade FLOPS for memory.

## Design
Single-op granularity, not segment-level checkpointing (`RematPass.cpp`):
for each cheap elementwise op (relu/gelu/sigmoid/add/mul/sub) with ≥2 uses
whose farthest use is more than `remat_threshold` ops away in program
order, every use past the nearest gets its own clone of the op planted
immediately before it — the original's live range shrinks to
`[def, nearest use]`, which the memory planning pass (step 7) then sees as
a smaller interval. This is the mechanical building block; true "recompute
the whole forward segment during backward" checkpointing needs a
forward/backward boundary marker this generic dialect-level pass doesn't
have — deferred to a training-loop-aware caller in Phase 6.

## Results
TODO: run on Linux with MLIR.

| Model | With remat: peak mem | Without: peak mem | Compute overhead |
|-------|---------------------|-------------------|-----------------|
| 12-layer transformer | TODO | TODO | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
