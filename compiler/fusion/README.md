# Operator Fusion Pass

**Status: STUB — requires MLIR on Linux.**

## What this measures
Pattern-matches fusable op sequences (matmul + bias_add + relu → fused_matmul_bias_relu)
and replaces them with a single fused op. Measures HBM traffic reduction.

## Results
TODO: run on Linux with MLIR.

| Pattern fused | HBM reads (before) | HBM reads (after) | Reduction |
|---------------|--------------------|--------------------|-----------|
| matmul+bias+relu | TODO | TODO | TODO |
| conv+bn+relu | TODO | TODO | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
