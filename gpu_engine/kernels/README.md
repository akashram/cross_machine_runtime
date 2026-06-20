# Element-wise GPU Kernels + GEMM

**Status: STUB — requires CUDA GPU. Run on g4dn.xlarge → p3.2xlarge for Tensor Cores.**

## What this measures
Element-wise kernels (add, mul, relu, gelu, softmax) optimized for coalescing and
occupancy, plus a GEMM progression: naive → shared-memory tiled → WMMA Tensor Core
→ cuBLAS baseline. Roofline analysis for each.

## Implementation notes

### Element-wise kernels
- All kernels: verify coalescing via coalescing_check.sh before benchmarking
- softmax: two-pass (max + sum) or online Welford — measure which is faster
- gelu: approximate (`x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))`) vs exact erf

### GEMM progression
1. **Naive**: one thread per output element, no reuse → bandwidth-bound
2. **Shared-memory tiled**: tile A and B into SRAM, reduce bank conflicts with padding
3. **WMMA (Tensor Cores)**: `wmma::load_matrix_sync` / `wmma::mma_sync` / `wmma::store_matrix_sync`
   - Requires M/N/K multiples of 16 (FP16) or 8 (TF32 on Ampere)
4. **cuBLAS**: `cublasSgemm` as the gold standard for comparison

## Results

TODO: run on GPU hardware and fill in this table.

### Element-wise throughput (GB/s)

| Kernel | N=1M | N=16M | N=256M | % of HBM peak |
|--------|-------|-------|--------|----------------|
| add | TODO | TODO | TODO | TODO |
| mul | TODO | TODO | TODO | TODO |
| relu | TODO | TODO | TODO | TODO |
| gelu | TODO | TODO | TODO | TODO |
| softmax | TODO | TODO | TODO | TODO |

### GEMM throughput (TFLOPS, M=N=K=4096, FP32)

| Variant | TFLOPS | % of peak |
|---------|--------|-----------|
| naive | TODO | TODO |
| tiled (shared mem) | TODO | TODO |
| WMMA (Tensor Core) | TODO | TODO |
| cuBLAS | TODO | TODO |

## Hardware notes
- Required: any CUDA GPU for element-wise + naive/tiled GEMM; Volta+ for WMMA
- Build preset: cuda (Linux)
- Tensor Core alignment: FP16 requires M/N/K % 16 == 0
