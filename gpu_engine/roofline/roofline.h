#pragma once
// GPU Roofline model — measure peak compute + peak bandwidth, then classify kernels.
//
// The Roofline model (Williams et al. 2009) characterizes kernel performance:
//   - Arithmetic Intensity (AI) = FLOP / bytes accessed
//   - If AI > ridge_point: compute-bound ceiling = peak_flops
//   - If AI < ridge_point: bandwidth-bound ceiling = AI × peak_bandwidth
//   - ridge_point = peak_flops / peak_bandwidth  (FLOPs per byte)
//
// Usage
// -----
//   RooflineHardware hw = measure_hardware();
//   print_roofline_header(hw);
//
//   // For each kernel to profile:
//   RooflineResult r = classify_kernel(achieved_tflops, flops_per_byte, hw);
//   print_roofline_result("my_kernel", r);
//
// Hardware measurement methods
// ----------------------------
//   measure_peak_flops_tflops(): large square cuBLAS SGEMM at M=N=K=4096,
//     repeated until stable.  Upper-bounds compute throughput.
//
//   measure_peak_bandwidth_gbs(): STREAM TRIAD (C = A + s*B) on 256 MB arrays.
//     Reports sustained HBM bandwidth (lower than spec due to ECC).

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

struct RooflineHardware {
    double peak_fp32_tflops;   // measured via cuBLAS SGEMM
    double peak_fp16_tflops;   // measured via cuBLAS HGEMM (Tensor Cores)
    double peak_bandwidth_gbs; // measured via STREAM TRIAD
    double ridge_point_fp32;   // = peak_fp32_tflops * 1e12 / (peak_bandwidth_gbs * 1e9)
    double ridge_point_fp16;
};

struct RooflineResult {
    double arithmetic_intensity; // FLOP/byte
    double achieved_tflops;
    double ceiling_tflops;       // min(peak_flops, AI × peak_bw)
    double utilization;          // achieved / ceiling
    bool   compute_bound;        // true if compute-bound
};

// ---- STREAM TRIAD kernel ------------------------------------------------

namespace detail {

__global__ void stream_triad(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + scalar * B[i];
}

} // namespace detail

// ---- Public API ---------------------------------------------------------

inline double measure_peak_bandwidth_gbs(int iters = 30) {
    // 256 MB per array × 3 arrays = 768 MB total per iteration
    const size_t N = (256ULL * 1024 * 1024) / sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    cudaMemset(d_A, 0, N * sizeof(float));
    cudaMemset(d_B, 1, N * sizeof(float));

    int threads = 256;
    dim3 grid((N + threads - 1) / threads);

    // Warmup
    for (int i = 0; i < 5; ++i)
        detail::stream_triad<<<grid, threads>>>(d_A, d_B, d_C, 2.0f, (int)N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i)
        detail::stream_triad<<<grid, threads>>>(d_A, d_B, d_C, 2.0f, (int)N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms; cudaEventElapsedTime(&ms, t0, t1);
    double elapsed_s = ms / 1e3 / iters;
    // TRIAD accesses: 2 reads (A, B) + 1 write (C)
    double bytes = 3.0 * N * sizeof(float);
    double bw_gbs = bytes / elapsed_s / 1e9;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return bw_gbs;
}

inline double measure_peak_flops_tflops(bool use_fp16 = false, int iters = 20) {
    const int S = 4096;
    cublasHandle_t handle;
    cublasCreate(&handle);

    if (use_fp16) {
        __half *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, (size_t)S * S * sizeof(__half));
        cudaMalloc(&d_B, (size_t)S * S * sizeof(__half));
        cudaMalloc(&d_C, (size_t)S * S * sizeof(__half));
        cudaMemset(d_A, 0, (size_t)S * S * sizeof(__half));
        cudaMemset(d_B, 0, (size_t)S * S * sizeof(__half));

        // Warmup
        const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
        for (int i = 0; i < 5; ++i)
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        S, S, S, &alpha_h, d_B, S, d_A, S, &beta_h, d_C, S);
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; ++i)
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        S, S, S, &alpha_h, d_B, S, d_A, S, &beta_h, d_C, S);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms; cudaEventElapsedTime(&ms, t0, t1);
        double tflops = 2.0 * S * S * S / (ms / 1e3 / iters) / 1e12;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cublasDestroy(handle);
        return tflops;
    } else {
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, (size_t)S * S * sizeof(float));
        cudaMalloc(&d_B, (size_t)S * S * sizeof(float));
        cudaMalloc(&d_C, (size_t)S * S * sizeof(float));
        cudaMemset(d_A, 0, (size_t)S * S * sizeof(float));
        cudaMemset(d_B, 0, (size_t)S * S * sizeof(float));

        float alpha = 1.0f, beta = 0.0f;
        for (int i = 0; i < 5; ++i)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        S, S, S, &alpha, d_B, S, d_A, S, &beta, d_C, S);
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; ++i)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        S, S, S, &alpha, d_B, S, d_A, S, &beta, d_C, S);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms; cudaEventElapsedTime(&ms, t0, t1);
        double tflops = 2.0 * S * S * S / (ms / 1e3 / iters) / 1e12;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cublasDestroy(handle);
        return tflops;
    }
}

inline RooflineHardware measure_hardware() {
    RooflineHardware hw;
    hw.peak_fp32_tflops   = measure_peak_flops_tflops(false);
    hw.peak_fp16_tflops   = measure_peak_flops_tflops(true);
    hw.peak_bandwidth_gbs = measure_peak_bandwidth_gbs();
    hw.ridge_point_fp32   = hw.peak_fp32_tflops * 1e12 / (hw.peak_bandwidth_gbs * 1e9);
    hw.ridge_point_fp16   = hw.peak_fp16_tflops * 1e12 / (hw.peak_bandwidth_gbs * 1e9);
    return hw;
}

inline RooflineResult classify_kernel(double achieved_tflops,
                                       double flops_per_byte,
                                       const RooflineHardware& hw,
                                       bool use_fp16 = false) {
    double peak_flops = use_fp16 ? hw.peak_fp16_tflops : hw.peak_fp32_tflops;
    double bw_ceiling = flops_per_byte * hw.peak_bandwidth_gbs / 1e3; // TFLOPS
    double ceiling    = (bw_ceiling < peak_flops) ? bw_ceiling : peak_flops;
    bool   compute_b  = (peak_flops <= bw_ceiling);

    RooflineResult r;
    r.arithmetic_intensity = flops_per_byte;
    r.achieved_tflops      = achieved_tflops;
    r.ceiling_tflops       = ceiling;
    r.utilization          = achieved_tflops / ceiling;
    r.compute_bound        = compute_b;
    return r;
}

inline void print_roofline_header(const RooflineHardware& hw) {
    printf("=== GPU Roofline ===\n");
    printf("  Peak FP32 TFLOPS : %.2f\n", hw.peak_fp32_tflops);
    printf("  Peak FP16 TFLOPS : %.2f\n", hw.peak_fp16_tflops);
    printf("  Peak BW   (GB/s) : %.2f\n", hw.peak_bandwidth_gbs);
    printf("  Ridge pt FP32    : %.2f FLOP/byte\n", hw.ridge_point_fp32);
    printf("  Ridge pt FP16    : %.2f FLOP/byte\n", hw.ridge_point_fp16);
    printf("\n");
}

inline void print_roofline_result(const char* name, const RooflineResult& r) {
    printf("  %-30s  AI=%-8.2f  %.2f / %.2f TFLOPS  util=%.1f%%  [%s]\n",
           name,
           r.arithmetic_intensity,
           r.achieved_tflops, r.ceiling_tflops,
           r.utilization * 100.0,
           r.compute_bound ? "compute-bound" : "bandwidth-bound");
}
