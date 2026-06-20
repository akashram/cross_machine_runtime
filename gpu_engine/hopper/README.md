# Hopper TMA + WGMMA

**Status: STUB — requires H100 (compute capability 9.0). Run on p5.48xlarge.**

## What this measures
- **TMA (Tensor Memory Accelerator)**: async bulk tensor copy using TMA descriptors
  (`cuda::memcpy_async` with TMA), measuring throughput vs `cudaMemcpyAsync`.
- **WGMMA (Warpgroup MMA)**: warpgroup-level matrix multiply, comparing against
  WMMA API (Volta/Ampere) in throughput (TFLOPS).

## Implementation notes

### TMA
- TMA descriptor: `CUtensorMap` — encodes tensor shape, strides, dtype, swizzle
- API: `cuda::memcpy_async` with `cuda::aligned_size_t` destination
- Advantage: overlaps data movement with computation, no warp divergence from memcpy
- Measure: sustained HBM bandwidth (GB/s) vs cudaMemcpyAsync for same tensor shape

### WGMMA
- 4 warps (128 threads) cooperate per MMA instruction
- `wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16` (example)
- Requires SM 9.0 — will not compile for earlier architectures
- Compare throughput vs WMMA (`wmma::mma_sync`) on same problem size

## Results

TODO: run on H100 hardware and fill in this table.

### TMA bandwidth

| Transfer size | TMA (GB/s) | cudaMemcpyAsync (GB/s) | Ratio |
|---------------|-----------|------------------------|-------|
| 1 MB | TODO | TODO | TODO |
| 64 MB | TODO | TODO | TODO |
| 1 GB | TODO | TODO | TODO |

### WGMMA vs WMMA throughput (FP16, M=N=K=4096)

| API | TFLOPS | % of H100 peak (989 TFLOPS FP16) |
|-----|--------|-----------------------------------|
| WMMA (sm_80 compat) | TODO | TODO |
| WGMMA (sm_90 native) | TODO | TODO |
| cuBLAS (sm_90) | TODO | TODO |

## Hardware notes
- Required: H100 — compute capability 9.0 (p5.48xlarge)
- Code is guarded with `#if __CUDA_ARCH__ >= 900`
- Build preset: cuda with `-arch=sm_90`
