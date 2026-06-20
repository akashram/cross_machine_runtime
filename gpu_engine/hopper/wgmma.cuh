#pragma once
// Hopper WGMMA (Warpgroup Matrix Multiply Accumulate) stubs
// TODO: implement on H100 hardware

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900

// WGMMA tile: 4 warps (128 threads) per warpgroup perform a collaborative MMA.
// Template: M=64, N=256, K=16, dtype=BF16 input, FP32 accumulate
template <int M, int N, int K>
__device__ void wgmma_mma_async_bf16_f32(
    float* acc,               // accumulator in registers (M*N/128 elements per thread)
    const __bfloat16* a_smem, // A tile in shared memory [M x K]
    const __bfloat16* b_smem  // B tile in shared memory [K x N]
);

#else
// Stub for non-Hopper
template <int M, int N, int K>
inline void wgmma_mma_async_bf16_f32(float*, const void*, const void*) {}
#endif
