// power_bench.cu — Step 23: GPU power monitoring via NVML.
//
// Measures power draw, temperature, and thermal throttle for:
//   - Idle GPU (no kernels running)
//   - Memory bandwidth kernel (STREAM TRIAD) — power scales with BW usage
//   - Compute kernel (cuBLAS SGEMM) — power scales with FP32 TFLOPS
//   - FP16 GEMM (Tensor Cores) — typically higher power than FP32
//
// Each workload runs for ~5 seconds to allow the power monitor to collect
// enough samples for a stable average.
//
// Expected results (A100 SXM, 400 W TDP):
//   Idle           ≈  50 W
//   STREAM TRIAD   ≈ 250 W
//   SGEMM FP32     ≈ 350 W
//   HGEMM FP16 TC  ≈ 400 W (may throttle briefly on boost clocks)
//
// Build: cmake --preset release (Linux, CUDA toolkit + NVML)
// Link: -lnvidia-ml

#define POWER_MONITOR_IMPL
#include "gpu_engine/power/power_monitor.h"

#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

// STREAM TRIAD kernel
__global__ void stream_triad(const float* A, const float* B, float* C,
                               float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + s * B[i];
}

// Measure power during a sustained workload by looping for ~target_seconds.
struct PowerBenchResult {
    PowerReport report;
    double      achieved_tflops_or_gbs; // context-dependent
};

static void run_for_seconds(double target_s, std::function<void()> launch) {
    auto t0 = std::chrono::high_resolution_clock::now();
    while (true) {
        launch();
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now - t0).count() >= target_s) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    const double DURATION_S = 5.0;
    PowerMonitor pm(0);

    // Print column header
    printf("%-24s  %-10s  %-10s  %-10s  %-8s  %s\n",
           "Workload", "Avg (W)", "Peak (W)", "Energy (J)", "MaxTemp", "Throttled?");
    printf("%s\n",
           "--------------------------------------------------------------------------");

    // ---- Idle -----------------------------------------------------------
    {
        pm.start();
        // Just sleep for the duration — no kernels
        auto t0 = std::chrono::high_resolution_clock::now();
        while (std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - t0).count() < DURATION_S) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        auto r = pm.stop();
        printf("%-24s  %-10.1f  %-10.1f  %-10.3f  %-8u  %s\n",
               "Idle", r.avg_power_w, r.peak_power_w,
               r.energy_j, r.max_temp_c, r.throttled ? "YES" : "no");
    }

    // ---- STREAM TRIAD ---------------------------------------------------
    {
        const int N = (256 * 1024 * 1024) / sizeof(float);
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_A, 0, (size_t)N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_B, 1, (size_t)N * sizeof(float)));

        dim3 block(256), grid((N + 255) / 256);
        // Warmup
        stream_triad<<<grid, block>>>(d_A, d_B, d_C, 2.0f, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        pm.start();
        run_for_seconds(DURATION_S, [&]{
            stream_triad<<<grid, block>>>(d_A, d_B, d_C, 2.0f, N);
        });
        auto r = pm.stop();
        printf("%-24s  %-10.1f  %-10.1f  %-10.3f  %-8u  %s\n",
               "STREAM TRIAD (256 MB)",
               r.avg_power_w, r.peak_power_w,
               r.energy_j, r.max_temp_c, r.throttled ? "YES" : "no");

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    // ---- SGEMM ----------------------------------------------------------
    {
        const int S = 4096;
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)S * S * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)S * S * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)S * S * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_A, 0, (size_t)S * S * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_B, 0, (size_t)S * S * sizeof(float)));

        float alpha = 1.0f, beta = 0.0f;
        // Warmup
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  S, S, S, &alpha, d_B, S, d_A, S, &beta, d_C, S));
        CUDA_CHECK(cudaDeviceSynchronize());

        pm.start();
        run_for_seconds(DURATION_S, [&]{
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      S, S, S, &alpha, d_B, S, d_A, S, &beta, d_C, S));
        });
        auto r = pm.stop();
        printf("%-24s  %-10.1f  %-10.1f  %-10.3f  %-8u  %s\n",
               "SGEMM FP32 4096³",
               r.avg_power_w, r.peak_power_w,
               r.energy_j, r.max_temp_c, r.throttled ? "YES" : "no");

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    // ---- HGEMM (Tensor Cores) ------------------------------------------
    if (prop.major >= 7) {
        const int S = 4096;
        __half *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)S * S * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)S * S * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)S * S * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_A, 0, (size_t)S * S * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_B, 0, (size_t)S * S * sizeof(__half)));

        const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
        // Warmup
        CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  S, S, S, &alpha_h, d_B, S, d_A, S, &beta_h, d_C, S));
        CUDA_CHECK(cudaDeviceSynchronize());

        pm.start();
        run_for_seconds(DURATION_S, [&]{
            CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      S, S, S, &alpha_h, d_B, S, d_A, S, &beta_h, d_C, S));
        });
        auto r = pm.stop();
        printf("%-24s  %-10.1f  %-10.1f  %-10.3f  %-8u  %s\n",
               "HGEMM FP16 TC 4096³",
               r.avg_power_w, r.peak_power_w,
               r.energy_j, r.max_temp_c, r.throttled ? "YES" : "no");

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    printf("\nNote: throttle_mask bits — see nvml.h NVML_CLOCKS_THROTTLE_REASON_*\n");
    printf("Common reasons: SW_PowerCap (0x4), HW_Thermal (0x1), HW_PowerBrake (0x2)\n");

    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
