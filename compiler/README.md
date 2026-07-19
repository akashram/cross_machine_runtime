# Phase 4: Compiler / IR (MLIR)

**Status: code-complete (15/15 steps), not yet built — requires LLVM/MLIR
built from source on Linux. Exception: `cost_model/` has no MLIR
dependency and has already been built and run — see its README.**

## Overview
MLIR-based compiler pipeline for the cross-machine runtime. Defines a custom
dialect with ops for matmul/conv/elementwise/reduce with device placement
annotations, then runs a series of passes to optimize and lower to
device-specific code. See `DESIGN.md` for the non-obvious decisions behind
the dialect and pass design.

## Build instructions

### 1. Build LLVM/MLIR from source
```bash
# On Linux x86 (any instance, c5.2xlarge is fine)
cd compiler/mlir_setup
./build_llvm.sh /path/to/llvm-project /path/to/build
```

### 2. Configure this project with MLIR
```bash
cmake -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release \
  -B build/compiler .
cmake --build build/compiler
```

## Pass pipeline (in order)

1. **mlir_setup** — LLVM build + CMake integration
2. **dialect** — RuntimeDialect: ops, types, assembly format, round-trip tests
3. **shape_inference** — propagate tensor shapes through op graph
4. **fusion** — fuse matmul+bias+relu → single fused op
5. **affine_lower** — lower loop nests to Affine dialect for polyhedral analysis
6. **mem_planning** — liveness analysis, buffer aliasing, peak memory minimization
7. **remat** — activation checkpointing via rematerialization
8. **placement** — assign ops to CPU/GPU/FPGA/TPU via cost model
9. **sharding** — GSPMD-style tensor sharding + comm op insertion
10. **kernel_spec** — lower dialect ops to device-specific kernels
11. **aot** — end-to-end: parse IR → optimize → lower → codegen → binary
12. **cost_model** — calibrated FLOPS/sec + bandwidth per device
13. **fuzzing** — libFuzzer for IR parser + pass pipeline
14. **upstream** — track LLVM bugs/patches found during development

## Hardware notes
- Required: Linux x86 (any AWS instance) for MLIR build
- LLVM build takes ~1 hour and ~30GB disk
- GPU backend lowering validation requires CUDA instance
- Exception: `cost_model/` builds and runs anywhere right now —
  `cmake -B build .` from the repo root picks it up unconditionally, no
  `MLIR_DIR` needed. Confirmed working via `clang++ -std=c++20` directly,
  see `cost_model/README.md`.
