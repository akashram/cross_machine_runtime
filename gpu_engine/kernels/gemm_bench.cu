// gemm_bench.cu — four-way GEMM comparison across square matrix sizes.
//
// Variants benchmarked:
//   naive          — one thread per C[i][j], no shared memory
//   tiled<16>      — 16×16 shared-memory tiling, 256 threads/block
//   tiled<32>      — 32×32 shared-memory tiling, 1024 threads/block
//   wmma           — Tensor Core 16×16×16 tiles (FP16 in, FP32 out)
//   cublas_sgemm   — cuBLAS FP32 (ground truth and performance ceiling)
//
// Metric: TFLOPS = 2*M*N*K / elapsed_seconds / 1e12
//
// Correctness: all variants verified against cuBLAS output with 1e-3 tolerance.

#include "gpu_engine/kernels/gemm.cuh"
#include <cublas_v2.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));          \
        exit(1);                                                      \
    }                                                                 \
} while (0)

#define CUBLAS_CHECK(call) do {                                       \
    cublasStatus_t _s = (call);                                       \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n",                \
                (int)_s, __FILE__, __LINE__);                         \
        exit(1);                                                      \
    }                                                                 \
} while (0)

template<typename F>
static float time_kernel_ms(F launch, int warmup = 5, int iters = 20) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

static double tflops(int M, int N, int K, float ms) {
    return 2.0 * M * N * K / (ms / 1e3) / 1e12;
}

// Convert float array to __half on device.
static void float_to_half(const float* d_f, __half* d_h, int n) {
    // Kernel to cast float → __half element-wise
    auto cast = [] __device__ (float v) { return __float2half(v); };
    // Use thrust or a simple kernel
    // Simple inline kernel via lambda capture isn't trivial; use cublas utility or a small kernel.
    // Write a small cast kernel:
    struct Caster {
        static __global__ void run(const float* f, __half* h, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) h[i] = __float2half(f[i]);
        }
    };
    int blocks = (n + 255) / 256;
    Caster::run<<<blocks, 256>>>(d_f, d_h, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Verify that two float arrays agree within `tol` (max absolute error).
static bool verify(const float* h_ref, const float* h_test, int n, float tol = 1e-2f) {
    float max_err = 0;
    for (int i = 0; i < n; ++i)
        max_err = fmaxf(max_err, fabsf(h_ref[i] - h_test[i]));
    printf("    max_abs_err = %.2e  (%s)\n", max_err, max_err <= tol ? "PASS" : "FAIL");
    return max_err <= tol;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    const bool has_tensor_cores = (prop.major >= 7);  // Volta+
    if (!has_tensor_cores)
        printf("Warning: no Tensor Cores (CC < 7.0); skipping WMMA variant.\n\n");

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Test sizes — all multiples of 64 for WMMA compatibility.
    // Skip naive for sizes > 1024 (too slow to meaningfully benchmark).
    const int sizes[] = {512, 1024, 2048, 4096};

    printf("%-6s  %-16s  %8s  %8s\n", "M=N=K", "variant", "ms", "TFLOPS");
    printf("%s\n", std::string(50, '-').c_str());

    for (int S : sizes) {
        const int M = S, N = S, K = S;
        const size_t sz_f = (size_t)M * N * sizeof(float);

        // --- Allocate device buffers ---
        float *d_A, *d_B, *d_C, *d_C_ref;
        CUDA_CHECK(cudaMalloc(&d_A,     (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B,     (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C,     sz_f));
        CUDA_CHECK(cudaMalloc(&d_C_ref, sz_f));

        // Fill A, B with small random-ish values to avoid FP overflow
        std::vector<float> h_A(M * K), h_B(K * N);
        for (int i = 0; i < M * K; ++i) h_A[i] = (i % 7 - 3) * 0.01f;
        for (int i = 0; i < K * N; ++i) h_B[i] = (i % 5 - 2) * 0.01f;
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

        // --- cuBLAS reference ---
        // Computes C = A × B (row-major) via: C^T = B^T × A^T (col-major trick)
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha, d_B, N,
                                         d_A, K,
                                 &beta,  d_C_ref, N));
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms;

        // Print size header
        printf("\nM=N=K=%d\n", S);

        // --- cuBLAS timing ---
        ms = time_kernel_ms([&]{
            cublasSgemm(cublas,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        });
        printf("  %-16s  %8.3f  %8.2f\n", "cublas_sgemm", ms, tflops(M, N, K, ms));

        // --- tiled<16> ---
        {
            dim3 block(16, 16), grid = gemm_tiled_grid<16>(M, N);
            ms = time_kernel_ms([&]{
                gemm_tiled<16><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            });
            printf("  %-16s  %8.3f  %8.2f", "tiled<16>", ms, tflops(M, N, K, ms));

            // Verify against cuBLAS
            std::vector<float> h_ref(M * N), h_test(M * N);
            CUDA_CHECK(cudaMemcpy(h_ref.data(),  d_C_ref, sz_f, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_test.data(), d_C,     sz_f, cudaMemcpyDeviceToHost));
            printf("\n");
            verify(h_ref.data(), h_test.data(), M * N);
        }

        // --- tiled<32> ---
        {
            dim3 block(32, 32), grid = gemm_tiled_grid<32>(M, N);
            ms = time_kernel_ms([&]{
                gemm_tiled<32><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            });
            printf("  %-16s  %8.3f  %8.2f\n", "tiled<32>", ms, tflops(M, N, K, ms));
        }

        // --- naive (small sizes only) ---
        if (S <= 1024) {
            dim3 block(16, 16), grid = gemm_naive_grid(M, N);
            ms = time_kernel_ms([&]{
                gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            }, /*warmup=*/3, /*iters=*/5);
            printf("  %-16s  %8.3f  %8.2f\n", "naive", ms, tflops(M, N, K, ms));
        }

        // --- WMMA (Volta+, fp16) ---
        if (has_tensor_cores) {
            __half *d_Ah, *d_Bh;
            CUDA_CHECK(cudaMalloc(&d_Ah, (size_t)M * K * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&d_Bh, (size_t)K * N * sizeof(__half)));
            float_to_half(d_A, d_Ah, M * K);
            float_to_half(d_B, d_Bh, K * N);

            dim3 block(WMMA_WARPS_PER_BLOCK * 32, 1);
            dim3 grid = gemm_wmma_grid(M, N);
            ms = time_kernel_ms([&]{
                gemm_wmma<<<grid, block>>>(d_Ah, d_Bh, d_C, M, N, K);
            });
            printf("  %-16s  %8.3f  %8.2f", "wmma (fp16->fp32)", ms, tflops(M, N, K, ms));

            // Verify — WMMA is fp16, expect ~1% error vs fp32 cuBLAS
            std::vector<float> h_ref(M * N), h_wmma(M * N);
            CUDA_CHECK(cudaMemcpy(h_ref.data(),  d_C_ref, sz_f, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_wmma.data(), d_C,     sz_f, cudaMemcpyDeviceToHost));
            printf("\n");
            verify(h_ref.data(), h_wmma.data(), M * N, /*tol=*/1e-1f);

            CUDA_CHECK(cudaFree(d_Ah));
            CUDA_CHECK(cudaFree(d_Bh));
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_C_ref));
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
