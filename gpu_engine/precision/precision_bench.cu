// precision_bench.cu — BF16 vs FP32 GEMM throughput and Tensor Core alignment cliff.
//
// Three benchmarks:
//
// 1. FP32 vs BF16 throughput
//    cuBLAS SGEMM (FP32) vs GEMM with BF16 inputs (GemmEx with CUDA_R_16BF).
//    BF16 Tensor Cores should be ~2× faster than FP32 TF32 Tensor Cores on A100.
//
// 2. LossScaler step simulation
//    Demonstrates the dynamic scale update logic (no GPU kernels — host only).
//    Simulates 10 clean steps + 1 overflow + recovery.
//
// 3. Tensor Core alignment cliff
//    Sweeps M from 1..64 with N=K=4096, plots TFLOPS.  At M=16, 32, 48, 64
//    (multiples of 8 for TF32), performance jumps vs M=15, 31, etc.

#include "gpu_engine/precision/mixed_precision.h"
#include "gpu_engine/precision/tensor_align.h"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
static float time_ms(F launch, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

static double tflops(int M, int N, int K, float ms) {
    return 2.0 * M * N * K / (ms / 1e3) / 1e12;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    const bool has_bf16 = (prop.major >= 8);  // Ampere+
    if (!has_bf16)
        printf("Warning: BF16 Tensor Cores require Ampere (CC 8.0+); "
               "FP32/TF32 results only.\n\n");

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // ---- 1. FP32 vs BF16 throughput ----------------------------------
    printf("=== GEMM throughput: FP32 (TF32 TC) vs BF16 ===\n");
    printf("  M=N=K: %4d  %4d  %4d  %4d\n", 1024, 2048, 4096, 8192);

    const int sizes[] = {1024, 2048, 4096, 8192};
    for (int S : sizes) {
        const int M = S, N = S, K = S;
        const size_t sz_f  = (size_t)M * N * sizeof(float);
        const size_t sz_bf = (size_t)M * N * sizeof(__bfloat16);

        float *d_Af, *d_Bf, *d_Cf;
        CUDA_CHECK(cudaMalloc(&d_Af, (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Bf, (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Cf, sz_f));

        __bfloat16 *d_Abf, *d_Bbf;
        float      *d_Cbf;
        if (has_bf16) {
            CUDA_CHECK(cudaMalloc(&d_Abf, (size_t)M * K * sizeof(__bfloat16)));
            CUDA_CHECK(cudaMalloc(&d_Bbf, (size_t)K * N * sizeof(__bfloat16)));
            CUDA_CHECK(cudaMalloc(&d_Cbf, sz_f));
            cast_fp32_to_bf16(d_Af, d_Abf, M * K);
            cast_fp32_to_bf16(d_Bf, d_Bbf, K * N);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        const float alpha = 1.0f, beta = 0.0f;

        float ms_fp32 = time_ms([&]{
            cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, d_Bf, N, d_Af, K, &beta, d_Cf, N);
        });

        float ms_bf16 = 0;
        if (has_bf16) {
            ms_bf16 = time_ms([&]{
                cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_Bbf, CUDA_R_16BF, N,
                             d_Abf, CUDA_R_16BF, K,
                             &beta,
                             d_Cbf, CUDA_R_32F,  N,
                             CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            });
        }

        printf("  M=N=K=%d: FP32=%.2f TFLOPS", S, tflops(M, N, K, ms_fp32));
        if (has_bf16)
            printf("  BF16=%.2f TFLOPS  (%.1f×)", tflops(M, N, K, ms_bf16),
                   ms_fp32 / ms_bf16);
        printf("\n");

        CUDA_CHECK(cudaFree(d_Af)); CUDA_CHECK(cudaFree(d_Bf)); CUDA_CHECK(cudaFree(d_Cf));
        if (has_bf16) {
            CUDA_CHECK(cudaFree(d_Abf)); CUDA_CHECK(cudaFree(d_Bbf)); CUDA_CHECK(cudaFree(d_Cbf));
        }
    }

    // ---- 2. LossScaler simulation ------------------------------------
    printf("\n=== LossScaler simulation ===\n");
    {
        LossScaler scaler(/*init=*/65536.0f, /*interval=*/5, 2.0f, 0.5f);
        for (int step = 0; step < 12; ++step) {
            bool overflow = (step == 7);  // simulate one overflow at step 7
            bool valid = scaler.step_and_update(overflow);
            printf("  step %2d: scale=%-12.0f  overflow=%-5s  step_valid=%s\n",
                   step, scaler.scale(),
                   overflow ? "true" : "false",
                   valid    ? "true" : "false");
        }
    }

    // ---- 3. Alignment cliff (TF32) -----------------------------------
    printf("\n=== Tensor Core alignment cliff (TF32, N=K=4096) ===\n");
    {
        // Sweep M from 1..40 — shows cliffs at multiples of 8 (TF32 alignment)
        auto pts = sweep_alignment_cliff(cublas,
                                         /*N=*/4096, /*K=*/4096,
                                         /*m_min=*/1, /*m_max=*/40,
                                         DType::TF32, /*iters=*/10);
        print_cliff_results(pts, DType::TF32, 4096, 4096);
        printf("\n  Tip: peaks at M=8, 16, 24, 32, 40 (multiples of alignment=8)\n");
        printf("       For BF16/FP16 the cliff is at multiples of 16.\n");
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
