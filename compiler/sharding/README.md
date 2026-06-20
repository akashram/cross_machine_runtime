# Auto-Sharding Pass (GSPMD-style)

**Status: STUB — requires MLIR on Linux. Multi-GPU validation requires p4d.**

## What this measures
GSPMD-style sharding: annotate tensors with sharding specs (replicated, split on dim N),
propagate specs through ops, insert all-gather/reduce-scatter where sharding changes.

## Results
TODO: run on Linux + multi-GPU hardware.

| Sharding config | Comm ops inserted | Correctness vs unsharded |
|----------------|------------------|--------------------------|
| 4-way data parallel | reduce-scatter + all-gather | TODO |
| 4-way tensor parallel (matmul) | all-reduce | TODO |

## Hardware notes
- Required: MLIR on Linux; multi-GPU for end-to-end validation
