# Shape Inference Pass

**Status: code-complete, not yet built — requires MLIR on Linux.**

## What this measures
Propagates tensor shapes through the op graph. Handles static shapes (constant
at compile time) and dynamic shapes (symbolic dimensions like batch size).

## Design
Worklist algorithm, same shape as the MLIR "Toy" tutorial's shape inference
pass: seed with every op whose result isn't a fully-static ranked tensor,
pop an op, call its `InferTypeOpInterface::inferReturnTypes`, and only
commit + re-enqueue users if the new type strictly refines the old one
(`isRefinement` in `ShapeInferencePass.cpp`). Iteration is capped at 4x the
initial worklist size so a genuinely-dynamic dim (e.g. batch size) causes a
warning and a dynamic result, not an infinite loop. Per-op inference logic
lives on the ops themselves (`RuntimeDialect.cpp`), not in this pass — see
`compiler/dialect/README.md`.

## Results
TODO: run on Linux with MLIR and fill in this table.

| Test | Result |
|------|--------|
| Static shape propagation through matmul chain | TODO |
| Dynamic batch dimension preserved as symbol | TODO |
| Shape conflict detection (mismatched dims) | TODO |

## Hardware notes
- Required: Linux x86 with MLIR built
