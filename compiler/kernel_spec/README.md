# Kernel Specialization Pass

**Status: code-complete, not yet built — requires MLIR on Linux.**

## What this measures
Lowers generic dialect ops to device-specific implementations:
- CPU: dispatch to AVX-512 kernels (cpu_engine/)
- GPU: dispatch to CUDA kernels (gpu_engine/)
- FPGA: dispatch to HLS kernels (fpga_engine/)

## Design
`(op-kind, device) -> symbol` and `(fusion_kind, device) -> symbol` lookup
tables in `KernelSpecPass.cpp`. Every op carrying a `runtime.device`
attribute (from placement, step 9) with a table entry is replaced by
`runtime.kernel_call`, and the callee symbol is declared as an external
`func.func private` in the module — a real, `SymbolTable`-verifiable
reference, not a bare string, so the AOT pipeline (step 12) can resolve it
against actual object files at link time. Ops with no table entry (a
kernel this project hasn't written yet, or a TPU target — Phase 8 lowers
through StableHLO instead of a direct kernel call) are left unlowered
rather than failing the pass; `runtime.kernels_lowered` /
`runtime.kernels_skipped` module attributes report the split.

## Results
TODO: run on Linux with MLIR.

| Op | Device | Lowered to |
|----|--------|-----------|
| runtime.matmul (CPU) | CPU | cpu_engine::avx512::matvec_f32 | 
| runtime.matmul (GPU) | GPU | gpu_engine::kernels::gemm_wmma |
| runtime.relu (FPGA) | FPGA | fpga_engine::dot_product |

## Hardware notes
- Required: Linux x86 with MLIR built
