# Kernel Specialization Pass

**Status: STUB — requires MLIR on Linux.**

## What this measures
Lowers generic dialect ops to device-specific implementations:
- CPU: dispatch to AVX-512 kernels (cpu_engine/)
- GPU: dispatch to CUDA kernels (gpu_engine/)
- FPGA: dispatch to HLS kernels (fpga_engine/)

## Results
TODO: run on Linux with MLIR.

| Op | Device | Lowered to |
|----|--------|-----------|
| runtime.matmul (CPU) | CPU | cpu_engine::avx512::matvec_f32 | 
| runtime.matmul (GPU) | GPU | gpu_engine::kernels::gemm_wmma |
| runtime.relu (FPGA) | FPGA | fpga_engine::dot_product |

## Hardware notes
- Required: Linux x86 with MLIR built
