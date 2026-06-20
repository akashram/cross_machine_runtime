# AOT Compilation Pipeline

**Status: STUB — requires MLIR on Linux.**

## What this measures
End-to-end pipeline: parse IR file → run all passes → lower to LLVM IR →
compile to native binary. Measures compilation time and validates generated
code quality vs hand-written kernels.

## Results
TODO: run on Linux with MLIR.

| Stage | Time (ms) | Notes |
|-------|----------|-------|
| Parse IR | TODO | TODO |
| Shape inference | TODO | TODO |
| Fusion | TODO | TODO |
| Placement | TODO | TODO |
| Codegen | TODO | TODO |
| Total compilation | TODO | TODO |
| Generated code vs hand-written (%) | TODO | TODO |

## Hardware notes
- Required: Linux x86 with MLIR + LLVM backend built
