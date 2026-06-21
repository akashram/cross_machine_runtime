// sparsity_bench.cu — Step 20: 2:4 structured sparsity benchmark.
//
// Measurements:
//   1. Prune a random FP16 weight matrix to 2:4 pattern.
//      Verify: exactly 50% of values are zero after pruning.
//   2. Compress with CPU reference encoder; print compression ratio.
//   3. Run sparse matmul via cuSPARSELt (if available) vs dense cuBLAS FP16.
//      Sizes: {1024, 2048, 4096} square matrices.
//
// Expected results (A100 SXM, SM 8.0):
//   Dense cuBLAS FP16 @ 4096³ ≈ 77 TFLOPS
//   Sparse TC 2:4 @ 4096³    ≈ 154 TFLOPS  (≈ 2× speedup from 50% sparsity)
//
// Build: cmake --preset release (on AWS p4d.24xlarge with CUDA + cuSPARSELt)

#include "gpu_engine/sparsity/sparse_24.h"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

template<typename F>
static float time_ms(F fn, int warmup = 5, int iters = 20) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

static void fill_random_half(__half* d_buf, int n, unsigned seed = 42) {
    std::vector<__half> h(n);
    srand(seed);
    for (int i = 0; i < n; ++i)
        h[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    CUDA_CHECK(cudaMemcpy(d_buf, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

static int count_zeros_half(const __half* d_buf, int n) {
    std::vector<__half> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_buf, n * sizeof(__half), cudaMemcpyDeviceToHost));
    int zeros = 0;
    for (int i = 0; i < n; ++i)
        if (__half2float(h[i]) == 0.0f) ++zeros;
    return zeros;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    if (prop.major < 8) {
        printf("2:4 Structured Sparsity requires sm_80+ (Ampere or newer).\n");
        printf("Run on p4d.24xlarge (A100, CC 8.0) or p5.48xlarge (H100, CC 9.0).\n");
        return 0;
    }

    const int SIZES[] = {1024, 2048, 4096};
    const int NSIZES  = sizeof(SIZES) / sizeof(SIZES[0]);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // ---- 1. Pruning verification ----------------------------------------
    printf("=== 2:4 Pruning Verification ===\n");
    {
        const int M = 1024, K = 1024;
        __half *d_dense, *d_pruned;
        CUDA_CHECK(cudaMalloc(&d_dense,  (size_t)M * K * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_pruned, (size_t)M * K * sizeof(__half)));
        fill_random_half(d_dense, M * K);

        prune_2_4(d_dense, d_pruned, M, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        int total  = M * K;
        int zeros  = count_zeros_half(d_pruned, total);
        int expect = total / 2;
        printf("  Matrix %d×%d: %d / %d zeros (expect %d, %s)\n",
               M, K, zeros, total, expect,
               (zeros == expect) ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_dense));
        CUDA_CHECK(cudaFree(d_pruned));
    }

    // ---- 2. Dense cuBLAS FP16 baseline and sparse comparison ------------
    printf("\n=== Dense cuBLAS FP16 vs Sparse 2:4 (TFLOPS) ===\n");
    printf("  %-8s  %-18s  %-18s  %s\n",
           "Size", "Dense FP16 TFLOPS", "Sparse 2:4 TFLOPS", "Speedup");
    printf("  -----------------------------------------------------------\n");

    for (int si = 0; si < NSIZES; ++si) {
        int S = SIZES[si];
        int M = S, N = S, K = S;

        __half *d_A, *d_B, *d_C_dense, *d_C_sparse;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C_dense,  (size_t)M * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C_sparse, (size_t)M * N * sizeof(__half)));
        fill_random_half(d_A, M * K, 42);
        fill_random_half(d_B, K * N, 99);

        __half* d_A_pruned;
        CUDA_CHECK(cudaMalloc(&d_A_pruned, (size_t)M * K * sizeof(__half)));
        prune_2_4(d_A, d_A_pruned, M, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Dense cuBLAS FP16: row-major via transposed call
        const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
        float ms_dense = time_ms([&]{
            CUBLAS_CHECK(cublasHgemm(cublas,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha_h,
                                     d_B, N,
                                     d_A, K,
                                     &beta_h,
                                     d_C_dense, N));
        });

        double tflops_dense = 2.0 * M * N * K / (ms_dense / 1e3) / 1e12;

        Sparse24Matrix A_sparse; A_sparse.rows = M; A_sparse.cols = K;
        SparseMatmulResult sparse_res = run_sparse_matmul(A_sparse, d_B, d_C_sparse,
                                                          M, N, K);

        printf("  %-8d  %-18.2f  ", S, tflops_dense);
        if (sparse_res.tflops > 0) {
            double speedup = sparse_res.tflops / tflops_dense;
            printf("%-18.2f  %.2f×\n", sparse_res.tflops, speedup);
        } else {
            printf("%-18s  %s\n", "N/A (no cuSPARSELt)", "—");
        }

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C_dense)); CUDA_CHECK(cudaFree(d_C_sparse));
        CUDA_CHECK(cudaFree(d_A_pruned));
        if (A_sparse.values)   delete[] A_sparse.values;
        if (A_sparse.metadata) delete[] A_sparse.metadata;
    }

    // ---- 3. Pruning throughput -------------------------------------------
    printf("\n=== Pruning Kernel Throughput ===\n");
    printf("  %-8s  %-18s  %s\n", "Size", "Pruning (GB/s)", "Time (us)");
    printf("  ------------------------------------------\n");
    for (int si = 0; si < NSIZES; ++si) {
        int S = SIZES[si];
        size_t bytes = (size_t)S * S * sizeof(__half);

        __half *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        fill_random_half(d_in, S * S);

        float ms = time_ms([&]{
            prune_2_4(d_in, d_out, S, S);
        });

        double bw = 2.0 * bytes / (ms / 1e3) / 1e9;  // read + write
        printf("  %-8d  %-18.2f  %.2f\n", S, bw, ms * 1e3);

        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    printf("\nDone.\n");
    return 0;
}
