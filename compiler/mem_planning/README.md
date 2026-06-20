# Memory Planning Pass

**Status: STUB — requires MLIR on Linux.**

## What this measures
Liveness analysis on tensor values to identify non-overlapping lifetimes,
then aliases buffers to minimize peak HBM usage. Compare vs XLA memory planning.

## Results
TODO: run on Linux with MLIR.

| Model | Naive peak mem (GB) | Planned peak mem (GB) | Reduction |
|-------|---------------------|-----------------------|-----------|
| MLP 4-layer | TODO | TODO | TODO |
| Transformer block | TODO | TODO | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
