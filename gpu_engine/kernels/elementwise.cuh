#pragma once
// Element-wise GPU kernels: add, mul, relu, gelu, softmax.
//
// Design notes
// ------------
// All kernels are 1D-indexed: grid covers N elements, each thread handles
// one element.  Blocks are fixed at 256 threads (good default occupancy).
//
// Softmax uses a two-pass strategy (separate max and normalisation passes)
// so the host can checkpoint between them if needed.  The reduce uses an
// inline warp + inter-warp reduction that mirrors block_reduce_sum in
// shared_ops.h but avoids a header dependency here.
//
// GELU: exact form (erff) matches PyTorch's default since 1.9.
// The fast tanh approximation (GPT-2 style) is provided as gelu_approx_kernel
// and is ~2× faster on older hardware where native erff is slow.

#include <cuda_runtime.h>
#include <math.h>   // erff, tanhf — available in device code

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// Warp-level max reduce: every lane gets the warp-wide maximum.
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    return v;
}

// Warp-level sum reduce: every lane gets the warp-wide sum.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Block-level max reduce using shared memory for inter-warp stage.
// smem must have at least ceil(blockDim.x / 32) floats.
__device__ __forceinline__ float block_max(float v, float* smem) {
    const int lane   = threadIdx.x % 32;
    const int wid    = threadIdx.x / 32;
    const int nwarps = (blockDim.x + 31) / 32;

    v = warp_reduce_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    v = (lane < nwarps) ? smem[lane] : -INFINITY;
    if (wid == 0) {
        v = warp_reduce_max(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// Block-level sum reduce using shared memory.
__device__ __forceinline__ float block_sum(float v, float* smem) {
    const int lane   = threadIdx.x % 32;
    const int wid    = threadIdx.x / 32;
    const int nwarps = (blockDim.x + 31) / 32;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    v = (lane < nwarps) ? smem[lane] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// -----------------------------------------------------------------------
// add, mul, relu, gelu
// -----------------------------------------------------------------------

__global__ void add_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void mul_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float*       __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void relu_kernel(const float* __restrict__ x,
                             float*       __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(0.0f, x[i]);
}

// Exact GELU: x · 0.5 · (1 + erf(x / √2))
// √2 = 1.41421356..., 1/√2 = 0.70710678...
__global__ void gelu_kernel(const float* __restrict__ x,
                             float*       __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v * 0.5f * (1.0f + erff(v * 0.70710678f));
    }
}

// Fast GELU approximation (GPT-2 / Hugging Face style, tanh-based).
// ~2× faster on older GPUs; converges to exact GELU within 0.02%.
// GELU_fast(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
__global__ void gelu_approx_kernel(const float* __restrict__ x,
                                    float*       __restrict__ y, int n) {
    constexpr float kSqrt2OverPi = 0.79788456f;
    constexpr float kCoeff       = 0.044715f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v  = x[i];
        float v3 = v * v * v;
        y[i] = 0.5f * v * (1.0f + tanhf(kSqrt2OverPi * (v + kCoeff * v3)));
    }
}

// -----------------------------------------------------------------------
// Softmax — two-pass, row-parallel
//
// Grid: one block per row (blockIdx.x = row index)
// Block: 256 threads iterating over columns in chunks
//
// Pass 1 (softmax_max_kernel):
//   Find row maximum; store to maxvals[row].
//
// Pass 2 (softmax_norm_kernel):
//   Compute S = Σ exp(x[i] - max); store normalised values.
//
// Both passes need at most 32 floats of shared memory (for inter-warp stage).
// -----------------------------------------------------------------------

__global__ void softmax_max_kernel(const float* __restrict__ x,
                                    float*       __restrict__ maxvals,
                                    int n, int rows) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + (long long)row * n;

    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        thread_max = fmaxf(thread_max, row_x[i]);

    __shared__ float smem[32];
    float row_max = block_max(thread_max, smem);

    if (threadIdx.x == 0) maxvals[row] = row_max;
}

__global__ void softmax_norm_kernel(const float* __restrict__ x,
                                     const float* __restrict__ maxvals,
                                     float*       __restrict__ y,
                                     int n, int rows) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + (long long)row * n;
    float*       row_y = y + (long long)row * n;
    float        mx    = maxvals[row];

    // Accumulate sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        thread_sum += expf(row_x[i] - mx);

    __shared__ float smem[32];
    float total = block_sum(thread_sum, smem);

    float inv_total = 1.0f / total;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        row_y[i] = expf(row_x[i] - mx) * inv_total;
}

// Convenience: launch both softmax passes.
// Caller must have allocated maxvals[rows] on device.
inline void launch_softmax(const float* x, float* maxvals, float* y,
                            int n, int rows,
                            int block_size = 256,
                            cudaStream_t stream = 0) {
    softmax_max_kernel <<<rows, block_size, 0, stream>>>(x, maxvals, n, rows);
    softmax_norm_kernel<<<rows, block_size, 0, stream>>>(x, maxvals, y, n, rows);
}
