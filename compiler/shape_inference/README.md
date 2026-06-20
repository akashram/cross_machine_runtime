# Shape Inference Pass

**Status: STUB — requires MLIR on Linux.**

## What this measures
Propagates tensor shapes through the op graph. Handles static shapes (constant
at compile time) and dynamic shapes (symbolic dimensions like batch size).

## Results
TODO: run on Linux with MLIR and fill in this table.

| Test | Result |
|------|--------|
| Static shape propagation through matmul chain | TODO |
| Dynamic batch dimension preserved as symbol | TODO |
| Shape conflict detection (mismatched dims) | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
