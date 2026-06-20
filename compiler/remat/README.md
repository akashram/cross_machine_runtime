# Rematerialization Pass (Activation Checkpointing)

**Status: STUB — requires MLIR on Linux.**

## What this measures
Identifies tensors cheaper to recompute than store (activations with low compute/memory ratio).
Inserts recompute ops at backward pass entry points to trade FLOPS for memory.

## Results
TODO: run on Linux with MLIR.

| Model | With remat: peak mem | Without: peak mem | Compute overhead |
|-------|---------------------|-------------------|-----------------|
| 12-layer transformer | TODO | TODO | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
