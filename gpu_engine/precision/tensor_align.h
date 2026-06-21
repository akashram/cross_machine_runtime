#pragma once
// Tensor Core alignment requirements and benchmark helper.
//
// Tensor Cores operate on fixed tile sizes.  If M, N, or K are not multiples
// of the required alignment, cuBLAS falls back to a non-Tensor-Core path and
// performance drops sharply — the "alignment cliff."
//
// Alignment requirements (CUDA 11.x, Ampere)
// -------------------------------------------
// FP16  → align 16  (WMMA 16×16×16 tile)
// BF16  → align 16
// TF32  → align 8   (cuBLAS uses 16×16×8 tiles internally)
// FP32  → no Tensor Core, any alignment works
// FP8   → align 16  (sm_90 Hopper)
//
// The cliff manifests as: TFLOPS for M=256 >> TFLOPS for M=255, because
// the misaligned case can't use the Tensor Core path.
//
// sweep_alignment_cliff() runs a GEMM sweep over M values from `m_min` to
// `m_max` and records TFLOPS at each, revealing the cliff.

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

enum class DType { FP16, BF16, TF32, FP32, FP8_E4M3, FP8_E5M2 };

inline int tensor_core_alignment(DType dtype) {
    switch (dtype) {
        case DType::FP8_E4M3: return 16;
        case DType::FP8_E5M2: return 16;
        case DType::FP16:     return 16;
        case DType::BF16:     return 16;
        case DType::TF32:     return 8;
        case DType::FP32:     return 1;
    }
    return 1;
}

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::FP8_E4M3: return "FP8_E4M3";
        case DType::FP8_E5M2: return "FP8_E5M2";
        case DType::FP16:     return "FP16";
        case DType::BF16:     return "BF16";
        case DType::TF32:     return "TF32";
        case DType::FP32:     return "FP32";
    }
    return "unknown";
}

inline void check_tensor_core_alignment(int M, int N, int K, DType dtype) {
    int align = tensor_core_alignment(dtype);
    if (align == 1) return;
    if (M % align != 0 || N % align != 0 || K % align != 0) {
        printf("WARNING: M=%d N=%d K=%d not multiples of %d (%s) — Tensor Core perf cliff!\n",
               M, N, K, align, dtype_name(dtype));
    }
}

// -----------------------------------------------------------------------
// Alignment cliff sweep — vary M from m_min..m_max, keep N=K fixed.
// Uses cuBLAS FP32 SGEMM (TF32 Tensor Cores on Ampere by default).
// To see the cliff: compare TFLOPS at M=aligned vs M=aligned-1.
// -----------------------------------------------------------------------

struct AlignPoint {
    int    M;
    double tflops;
    bool   aligned;  // M is a multiple of tensor_core_alignment(dtype)
};

inline std::vector<AlignPoint> sweep_alignment_cliff(
    cublasHandle_t cublas,
    int N, int K,
    int m_min, int m_max,
    DType dtype = DType::TF32,
    int iters = 20)
{
    // Allocate max-size buffers up front
    const size_t max_M = m_max;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, max_M * K * sizeof(float));
    cudaMalloc(&d_B, K * N     * sizeof(float));
    cudaMalloc(&d_C, max_M * N * sizeof(float));
    cudaMemset(d_A, 0, max_M * K * sizeof(float));
    cudaMemset(d_B, 0, K * N     * sizeof(float));

    int align = tensor_core_alignment(dtype);
    const float alpha = 1.0f, beta = 0.0f;

    std::vector<AlignPoint> results;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    for (int M = m_min; M <= m_max; ++M) {
        // Warm up
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaDeviceSynchronize();

        cudaEventRecord(t0);
        for (int i = 0; i < iters; ++i)
            cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);

        AlignPoint p;
        p.M       = M;
        p.tflops  = 2.0 * M * N * K / (ms / iters / 1e3) / 1e12;
        p.aligned = (M % align == 0);
        results.push_back(p);
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return results;
}

inline void print_cliff_results(const std::vector<AlignPoint>& pts,
                                 DType dtype, int N, int K) {
    int align = tensor_core_alignment(dtype);
    printf("\nAlignment cliff sweep: %s  N=%d K=%d  alignment=%d\n",
           dtype_name(dtype), N, K, align);
    printf("%-6s  %9s  %s\n", "M", "TFLOPS", "aligned?");
    printf("%s\n", std::string(28, '-').c_str());
    for (auto& p : pts) {
        printf("%-6d  %9.2f  %s\n", p.M, p.tflops,
               p.aligned ? "YES" : "   (cliff)");
    }
}
