# MLIR Build Setup

**Status: script complete, not yet run — requires Linux x86. Any AWS instance (c5.2xlarge recommended).**

## Design
`build_llvm.sh` configures LLVM/MLIR+Clang out-of-tree with Ninja,
`-DLLVM_ENABLE_PROJECTS="mlir;clang"` (Clang is needed for the fuzzing
step's `-fsanitize=fuzzer` support, not just MLIR itself),
`-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"` (X86 for the host, NVPTX so a future
GPU-targeting lowering path — out of scope for Phase 4 itself, which
dispatches to `gpu_engine/`'s existing CUDA kernels via `kernel_call` — has
the backend available if ever needed), and runs `check-mlir` so a broken
build fails at setup time rather than silently at the first real use.

## What this measures
Validates that LLVM/MLIR builds correctly from source, MLIR CMake integration
works, and the project can find all required dialects and tools.

## Results

TODO: run on Linux and fill in this table.

| Metric | Value |
|--------|-------|
| LLVM commit used | TODO |
| Build time (Ninja, 8 cores) | TODO |
| Disk usage (build dir) | TODO |
| `mlir-opt --version` | TODO |

## Hardware notes
- Required: Linux x86, ≥ 16GB RAM, ≥ 40GB disk
- Build preset: release (Linux)
- Recommended: c5.4xlarge (16 vCPU) to cut build time to ~30 min
