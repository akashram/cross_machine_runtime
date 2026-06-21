// elementwise_bench.cu — throughput benchmark for add/mul/relu/gelu/softmax.
//
// Metric: effective memory bandwidth (GB/s).
//   add:   reads 2×N×4B, writes N×4B  → 3×N×4B
//   mul:   reads 2×N×4B, writes N×4B  → 3×N×4B
//   relu:  reads N×4B, writes N×4B    → 2×N×4B
//   gelu:  reads N×4B, writes N×4B    → 2×N×4B
//   softmax (per row): 2 passes × (read + write) — reported as rows/sec
//
// Run with Nsight Compute for L2/HBM bandwidth to compare against roofline.

#include "gpu_engine/kernels/elementwise.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));          \
        exit(1);                                                      \
    }                                                                 \
} while (0)

// Returns the average kernel time over `iters` iterations in milliseconds.
// Runs `warmup` iterations first (not timed).
template<typename F>
static float time_kernel(F launch, int warmup = 10, int iters = 50) {
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

static void print_bw(const char* name, float ms, size_t bytes) {
    double gb  = bytes / 1e9;
    double sec = ms / 1e3;
    printf("  %-24s  %7.3f ms   %7.2f GB/s\n", name, ms, gb / sec);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  peak HBM BW = %.1f GB/s\n\n",
           prop.name,
           2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9);

    // ---- Element-wise benchmarks (1D) ----------------------------------
    constexpr int N       = 1 << 25;   // 32M floats = 128 MB — bandwidth-bound
    constexpr int THREADS = 256;
    const     int BLOCKS  = (N + THREADS - 1) / THREADS;

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_a, 0x3f, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0x3f, (size_t)N * sizeof(float)));

    printf("=== Element-wise kernels (N=%d, %.0f MB) ===\n",
           N, N * 4.0 / (1 << 20));
    printf("  %-24s  %9s   %9s\n", "kernel", "time", "bandwidth");
    printf("  %s\n", std::string(50, '-').c_str());

    {
        float ms = time_kernel([&]{ add_kernel<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N); });
        print_bw("add", ms, 3ULL * N * sizeof(float));
    }
    {
        float ms = time_kernel([&]{ mul_kernel<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N); });
        print_bw("mul", ms, 3ULL * N * sizeof(float));
    }
    {
        float ms = time_kernel([&]{ relu_kernel<<<BLOCKS, THREADS>>>(d_a, d_c, N); });
        print_bw("relu", ms, 2ULL * N * sizeof(float));
    }
    {
        float ms = time_kernel([&]{ gelu_kernel<<<BLOCKS, THREADS>>>(d_a, d_c, N); });
        print_bw("gelu (exact erff)", ms, 2ULL * N * sizeof(float));
    }
    {
        float ms = time_kernel([&]{ gelu_approx_kernel<<<BLOCKS, THREADS>>>(d_a, d_c, N); });
        print_bw("gelu (fast tanh)", ms, 2ULL * N * sizeof(float));
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // ---- Softmax benchmark (2D) ----------------------------------------
    // rows × cols matrix; each row is independently softmax-ed.
    // Two representative shapes: wide rows (attention scores), narrow rows.
    printf("\n=== Softmax (rows × cols) ===\n");
    printf("  %-24s  %9s\n", "shape", "time");
    printf("  %s\n", std::string(36, '-').c_str());

    struct SoftmaxCase { int rows; int cols; };
    const SoftmaxCase cases[] = {
        {1024,   512},   // small rows
        {1024,  2048},   // typical attention head
        {4096,  4096},   // large square
        {   1, 1<<20},   // single very long row
    };

    for (auto& tc : cases) {
        size_t total = (size_t)tc.rows * tc.cols;
        float *dx, *dy, *dmaxvals;
        CUDA_CHECK(cudaMalloc(&dx,       total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy,       total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dmaxvals, (size_t)tc.rows * sizeof(float)));
        CUDA_CHECK(cudaMemset(dx, 0x3f, total * sizeof(float)));

        float ms = time_kernel([&]{
            launch_softmax(dx, dmaxvals, dy, tc.cols, tc.rows);
        });
        printf("  %-24s  %7.3f ms\n",
               (std::to_string(tc.rows) + " × " + std::to_string(tc.cols)).c_str(), ms);

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
        CUDA_CHECK(cudaFree(dmaxvals));
    }

    // ---- Numerical correctness check -----------------------------------
    printf("\n=== Correctness check (softmax, 4 rows × 8 cols) ===\n");
    {
        constexpr int ROWS = 4, COLS = 8;
        float h_x[ROWS * COLS], h_y[ROWS * COLS];
        for (int i = 0; i < ROWS * COLS; ++i) h_x[i] = (float)(i % COLS) - 3.5f;

        float *dx, *dy, *dmx;
        CUDA_CHECK(cudaMalloc(&dx,  ROWS * COLS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy,  ROWS * COLS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dmx, ROWS         * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dx, h_x, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
        launch_softmax(dx, dmx, dy, COLS, ROWS);
        CUDA_CHECK(cudaMemcpy(h_y, dy, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        bool ok = true;
        for (int r = 0; r < ROWS; ++r) {
            float sum = 0;
            for (int c = 0; c < COLS; ++c) sum += h_y[r * COLS + c];
            if (fabsf(sum - 1.0f) > 1e-5f) { ok = false; break; }
        }
        printf("  Row sums = 1.0: %s\n", ok ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
        CUDA_CHECK(cudaFree(dmx));
    }

    return 0;
}
