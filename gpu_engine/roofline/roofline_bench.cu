// roofline_bench.cu — Step 21: GPU Roofline model.
//
// Measures peak FP32 and FP16 compute (via cuBLAS SGEMM/HGEMM) and peak HBM
// bandwidth (via STREAM TRIAD), then classifies a set of representative kernels
// against both ceilings.
//
// Kernels profiled:
//   - elementwise add (bandwidth-bound, AI ≈ 0.33 FLOP/byte)
//   - GEMM N=K=M=4096 (compute-bound with TCs)
//   - GEMM N=K=M=256  (bandwidth-bound, too small for TC saturation)
//   - softmax row-parallel (bandwidth-bound)
//
// Expected results (A100 SXM):
//   Peak FP32       ≈ 19.5 TFLOPS
//   Peak FP16 (TC)  ≈ 77   TFLOPS
//   Peak BW         ≈ 1800  GB/s
//   Ridge pt FP16   ≈ 42.8  FLOP/byte

#include "gpu_engine/roofline/roofline.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

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

// Simple elementwise add: C = A + B
__global__ void vec_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// Row-parallel softmax max pass (same as in elementwise.cuh)
__global__ void softmax_max(const float* x, float* maxvals, int cols, int rows) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = x + r * cols;
    float m = -1e38f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        m = fmaxf(m, row[c]);
    __shared__ float smem[32];
    // warp reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, mask));
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = (threadIdx.x < (blockDim.x / 32)) ? smem[threadIdx.x] : -1e38f;
        for (int mask = 16; mask > 0; mask >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, mask));
        if (threadIdx.x == 0) maxvals[r] = m;
    }
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // ---- 1. Measure hardware ceilings -----------------------------------
    printf("Measuring hardware ceilings...\n");
    RooflineHardware hw = measure_hardware();
    print_roofline_header(hw);

    // ---- 2. Classify kernels -------------------------------------------
    printf("=== Kernel Roofline Classification ===\n");
    printf("  %-30s  %-10s  %-20s  %-6s  %s\n",
           "Kernel", "AI (F/B)", "Perf/Ceiling TFLOPS", "Util%", "Bound");
    printf("  %s\n",
           "------------------------------------------------------------------------------------");

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // --- Elementwise add ---
    {
        const int N = 1 << 25;  // 32M elements
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_A, 0, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_B, 0, N * sizeof(float)));

        int threads = 256;
        dim3 grid((N + threads - 1) / threads);
        float ms = time_ms([&]{
            vec_add<<<grid, threads>>>(d_A, d_B, d_C, N);
        });

        // FLOPs = N additions; bytes = 3×N×4 (2 reads + 1 write)
        double flops       = (double)N;
        double bytes       = 3.0 * N * sizeof(float);
        double achieved_tf = flops / (ms / 1e3) / 1e12;
        double ai          = flops / bytes;

        auto r = classify_kernel(achieved_tf, ai, hw, false);
        print_roofline_result("vec_add (32M fp32)", r);

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    // --- GEMM large (should saturate Tensor Cores) ---
    for (int S : {4096, 2048, 256}) {
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)S * S * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)S * S * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)S * S * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_A, 0, (size_t)S * S * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_B, 0, (size_t)S * S * sizeof(float)));

        float alpha = 1.0f, beta = 0.0f;
        float ms = time_ms([&]{
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     S, S, S, &alpha, d_B, S, d_A, S, &beta, d_C, S));
        });

        double flops       = 2.0 * S * S * S;
        double bytes       = (2.0 * S * S + S * S) * sizeof(float);  // A, B reads + C write
        double achieved_tf = flops / (ms / 1e3) / 1e12;
        double ai          = flops / bytes;

        char label[64];
        snprintf(label, sizeof(label), "SGEMM %d×%d×%d", S, S, S);
        auto r = classify_kernel(achieved_tf, ai, hw, false);
        print_roofline_result(label, r);

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    // --- FP16 GEMM (Tensor Cores) ---
    {
        const int S = 4096;
        __half *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, (size_t)S * S * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B, (size_t)S * S * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C, (size_t)S * S * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_A, 0, (size_t)S * S * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_B, 0, (size_t)S * S * sizeof(__half)));

        const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
        float ms = time_ms([&]{
            CUBLAS_CHECK(cublasHgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     S, S, S, &alpha_h, d_B, S, d_A, S, &beta_h, d_C, S));
        });

        double flops       = 2.0 * S * S * S;
        double bytes       = (3.0 * S * S) * sizeof(__half);
        double achieved_tf = flops / (ms / 1e3) / 1e12;
        double ai          = flops / bytes;

        auto r = classify_kernel(achieved_tf, ai, hw, true);
        print_roofline_result("HGEMM TC 4096³", r);

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    // --- Softmax max-pass ---
    {
        const int ROWS = 4096, COLS = 4096;
        float *d_x, *d_maxvals;
        CUDA_CHECK(cudaMalloc(&d_x,       (size_t)ROWS * COLS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_maxvals,  ROWS * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_x, 0, (size_t)ROWS * COLS * sizeof(float)));

        float ms = time_ms([&]{
            softmax_max<<<ROWS, 256>>>(d_x, d_maxvals, COLS, ROWS);
        });

        // comparisons per element ≈ 1 FMA (approximate)
        double flops       = (double)ROWS * COLS;
        double bytes       = (double)ROWS * COLS * sizeof(float);  // read only
        double achieved_tf = flops / (ms / 1e3) / 1e12;
        double ai          = flops / bytes;

        auto r = classify_kernel(achieved_tf, ai, hw, false);
        print_roofline_result("softmax max 4096×4096", r);

        CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_maxvals));
    }

    // ---- 3. Summary -----------------------------------------------------
    printf("\n");
    printf("Ridge point FP32: %.1f FLOP/byte — below this is bandwidth-bound\n",
           hw.ridge_point_fp32);
    printf("Ridge point FP16: %.1f FLOP/byte — above this, Tensor Cores dominate\n",
           hw.ridge_point_fp16);
    printf("\nNote: Run `ncu --set full` on individual kernels to get precise AI\n"
           "  from l1tex__t_bytes and sm__sass_thread_inst_executed metrics.\n");

    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
