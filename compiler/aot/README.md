# AOT Compilation Pipeline

**Status: code-complete, not yet built — requires MLIR on Linux.**

## What this measures
End-to-end pipeline: parse IR file → run all passes → lower to LLVM IR →
compile to native binary. Measures compilation time and validates generated
code quality vs hand-written kernels.

## Design
`AotCompiler::compile()` (`AotCompiler.cpp`) runs, in order: parse →
shape-inference → fusion → affine-lower/tile → mem-planning → remat →
placement → sharding → kernel-spec → standard MLIR-to-LLVM-dialect lowering
(affine→scf→cf→llvm, plus arith/memref/func→llvm) → `translateModuleToLLVMIR`
→ object emission → link against `CompilerOptions::link_libraries` (the
cpu_engine/gpu_engine/fpga_engine `.a` files the placed-and-specialized IR's
`kernel_call` symbols resolve against). Every stage is wrapped in a
`std::chrono::steady_clock` timer into `CompileStats`; `aotc_main.cpp` is a
CLI (`runtime_aotc <in.mlir> <out> [--devices N] [--tile-size N]
[--remat-threshold F] [--link lib.a]`) that prints the stats table directly
in the format below — running it on Linux and pasting the output is the
entire "fill in this README" step. Failures are non-fatal per-stage (each
pass runs even if an earlier one partially failed) so a broken IR still
produces a useful timing/skip-count breakdown instead of an opaque abort.

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
