// coalescing_test.cu — contrasting good vs. bad global memory access patterns.
//
// Two kernels over an N-element float array:
//
//   coalesced_kernel   — thread i reads A[i], writes B[i].
//     Within a warp, 32 consecutive threads access 32 consecutive floats (128 bytes).
//     That maps to 4 × 32-byte L1 sectors → minimum possible sectors/request.
//
//   uncoalesced_kernel — thread i reads A[(i * STRIDE) % N], writes B[i].
//     With STRIDE = N/32, each thread in a warp is N/32 floats = N/8 bytes apart.
//     Each thread falls in a different L1 sector → maximum sectors/request.
//
// Purpose: pass this binary to coalescing_check.sh to verify the validator works
// and to measure the baseline ideal_ratio for coalesced_kernel on this GPU.
//
// Usage:
//   ./coalescing_test                          # dry run, no profiling
//   ./coalescing_check.sh ./coalescing_test    # run through validator
//   ncu --set full --csv ./coalescing_test     # detailed Nsight profile

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(_e));          \
        exit(1);                                                      \
    }                                                                 \
} while (0)

// -----------------------------------------------------------------------
// coalesced_kernel
// Each thread reads and writes its own contiguous element.
// Warp accesses: A[base], A[base+1], …, A[base+31] → 128B = 4 sectors.
// -----------------------------------------------------------------------
__global__ void coalesced_kernel(const float* __restrict__ A,
                                  float* __restrict__ B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) B[idx] = A[idx] * 2.0f;
}

// -----------------------------------------------------------------------
// uncoalesced_kernel
// Thread i reads A[(i * stride) % N].  With stride = N / 32, threads in
// the same warp are stride floats apart → each in a different 32B sector.
// Warp accesses: up to 32 distinct sectors for 1 warp request.
// -----------------------------------------------------------------------
__global__ void uncoalesced_kernel(const float* __restrict__ A,
                                    float* __restrict__ B, int N, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Cast to avoid overflow on large N
        int src = static_cast<int>((static_cast<long long>(tid) * stride) % N);
        B[tid] = A[src] * 2.0f;
    }
}

int main() {
    // N must be a multiple of 32 so the stride trick produces clean strided patterns.
    constexpr int N       = 1 << 22;  // 4M floats = 16 MB — fits comfortably in T4/V100 HBM
    constexpr int THREADS = 256;
    constexpr int BLOCKS  = N / THREADS;
    const     int STRIDE  = N / 32;   // worst-case: each warp thread in its own sector

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_A, 0x3f, (size_t)N * sizeof(float)));  // fill with small positive floats

    // Warm-up (not profiled by ncu unless --replay is set)
    coalesced_kernel  <<<BLOCKS, THREADS>>>(d_A, d_B, N);
    uncoalesced_kernel<<<BLOCKS, THREADS>>>(d_A, d_B, N, STRIDE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Profiled iterations — ncu captures these
    for (int i = 0; i < 5; ++i)
        coalesced_kernel<<<BLOCKS, THREADS>>>(d_A, d_B, N);
    for (int i = 0; i < 5; ++i)
        uncoalesced_kernel<<<BLOCKS, THREADS>>>(d_A, d_B, N, STRIDE);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("coalescing_test complete.\n");
    printf("  N=%d floats (%.1f MB),  STRIDE=%d,  blocks=%d,  threads/block=%d\n",
           N, N * 4.0 / (1<<20), STRIDE, BLOCKS, THREADS);
    printf("\nProfile command:\n");
    printf("  ./coalescing_check.sh ./coalescing_test\n");
    printf("Or for full Nsight detail:\n");
    printf("  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,"
               "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum --csv ./coalescing_test\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    return 0;
}
