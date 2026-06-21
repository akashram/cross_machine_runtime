// graph_bench.cu — CUDA Graph capture/replay vs eager launch overhead.
//
// Workload: chain of CHAIN_LEN simple element-wise kernels (add, each
// depending on the output of the previous).
//
// Eager path: submit CHAIN_LEN kernel launches via cudaLaunchKernel.
// Graph path: capture once, replay via a single cudaGraphLaunch.
//
// Expected: graph replay reduces CPU launch overhead by ~CHAIN_LEN× since
// the driver schedules all CHAIN_LEN nodes from a single submission.
//
// Metrics:
//   eager_us  — CPU µs per eager iteration (CHAIN_LEN launches)
//   graph_us  — CPU µs per graph iteration (1 launch)
//   speedup   — eager_us / graph_us  (expected: ~CHAIN_LEN)
//   GPU time  — wall-clock GPU execution (should be identical for both)

#include "gpu_engine/graphs/graph_runner.h"
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

// Simple kernel: out[i] = in[i] + val
__global__ void add_scalar(const float* in, float* out, float val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] + val;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n\n", prop.name);

    constexpr int N         = 1 << 20;  // 1M floats per buffer
    constexpr int THREADS   = 256;
    constexpr int BLOCKS    = N / THREADS;
    constexpr int CHAIN_LEN = 20;       // kernels per iteration
    constexpr int ITERS     = 2000;     // timing iterations

    // Ping-pong buffers: each kernel reads from buf[i%2] and writes to buf[(i+1)%2]
    float *d_buf[2];
    CUDA_CHECK(cudaMalloc(&d_buf[0], N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_buf[1], N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_buf[0], 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_buf[1], 0, N * sizeof(float)));

    // Create a non-default stream for graph capture
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // The workload: CHAIN_LEN sequential kernels (each depends on previous output)
    auto workload = [&]() {
        for (int k = 0; k < CHAIN_LEN; ++k)
            add_scalar<<<BLOCKS, THREADS, 0, stream>>>(
                d_buf[k % 2], d_buf[(k + 1) % 2], 1.0f, N);
    };

    // ---- Capture -------------------------------------------------------
    GraphRunner runner;
    runner.capture(stream, workload);
    runner.print_info();

    // ---- Measure overhead ----------------------------------------------
    auto result = runner.measure_overhead(stream, workload, ITERS);

    printf("Workload: %d sequential kernels, N=%d floats per kernel\n\n",
           CHAIN_LEN, N);
    printf("  %-20s  %8.2f µs/iter\n", "eager (CPU)",   result.eager_launch_us);
    printf("  %-20s  %8.2f µs/iter\n", "graph (CPU)",   result.graph_replay_us);
    printf("  %-20s  %8.2f×\n",        "speedup",       result.speedup);

    // ---- GPU execution time (should be equal for both) ----------------
    // Use CUDA events to measure actual GPU wall time.
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Eager GPU time
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < 100; ++i) workload();
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float eager_gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&eager_gpu_ms, t0, t1));

    // Graph GPU time
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < 100; ++i) runner.replay(stream);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float graph_gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&graph_gpu_ms, t0, t1));

    printf("\n  %-20s  %8.3f ms/iter  (100-iter avg)\n",
           "eager (GPU)",  eager_gpu_ms / 100.0f);
    printf("  %-20s  %8.3f ms/iter  (100-iter avg)\n",
           "graph (GPU)",  graph_gpu_ms / 100.0f);
    printf("  GPU time ratio: %.2f×  (should be ~1.0 — same work)\n",
           graph_gpu_ms / eager_gpu_ms);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_buf[0]));
    CUDA_CHECK(cudaFree(d_buf[1]));
    return 0;
}
