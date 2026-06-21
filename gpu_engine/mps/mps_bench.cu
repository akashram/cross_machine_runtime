// mps_bench.cu — Step 22: CUDA Multi-Process Service (MPS) benchmarking.
//
// CUDA MPS funnels multiple processes through a single context, eliminating
// per-process context switch overhead and enabling fine-grained GPU sharing.
//
// Without MPS: each process has an independent context; the GPU time-slices.
//   - Context switches can add 10–100 µs per kernel submission.
//   - Processes can't overlap unless they use streams within one context.
//
// With MPS: all processes share one context via the MPS server.
//   - Kernel submissions merge into the HW scheduler simultaneously.
//   - Throughput increases when many small kernels from N processes are pipelined.
//   - Latency per kernel can decrease significantly.
//
// This benchmark measures throughput and latency with/without MPS by:
//   1. Forking N child processes, each running a workload of K small kernels.
//   2. Parent measures elapsed wall time for all children to complete.
//   3. User runs once without MPS (DEFAULT mode), then enables MPS via
//      setup_mps.sh and runs again.
//
// Usage:
//   ./mps_bench [n_procs] [n_kernels]
//   ./mps_bench           → n_procs=4, n_kernels=1000
//
// Compare:
//   (No MPS)  nvidia-smi -i 0 -c DEFAULT  &&  ./mps_bench
//   (With MPS) sudo bash setup_mps.sh     &&  ./mps_bench

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[pid=%d] CUDA error %s:%d — %s\n", \
                (int)getpid(), __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// Trivial kernel — small workload per launch to amplify context-switch cost
__global__ void dummy_compute(float* buf, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] += val;
}

// Each child process runs this workload
static void run_child_workload(int n_kernels, int buf_n, double* out_ms) {
    CUDA_CHECK(cudaSetDevice(0));

    float* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, buf_n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_buf, 0, buf_n * sizeof(float)));

    // Warmup
    dim3 block(128);
    dim3 grid((buf_n + 127) / 128);
    for (int i = 0; i < 10; ++i)
        dummy_compute<<<grid, block>>>(d_buf, buf_n, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_kernels; ++i)
        dummy_compute<<<grid, block>>>(d_buf, buf_n, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    *out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CUDA_CHECK(cudaFree(d_buf));
}

// Write result to a shared-memory or pipe fd
static void write_result(int fd, double ms) {
    write(fd, &ms, sizeof(double));
}

int main(int argc, char** argv) {
    int n_procs   = (argc > 1) ? atoi(argv[1]) : 4;
    int n_kernels = (argc > 2) ? atoi(argv[2]) : 1000;
    const int BUF_N = 4096;  // small buffer so kernel is fast

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("n_procs=%d  n_kernels=%d  buf_n=%d\n\n", n_procs, n_kernels, BUF_N);

    if (prop.major < 3) {
        printf("MPS requires sm_35+. Exiting.\n");
        return 0;
    }

    // Check if running on Linux (MPS is Linux-only)
#ifndef __linux__
    printf("MPS is only available on Linux. This binary still benchmarks\n"
           "multi-process overhead without MPS to serve as a baseline.\n\n");
#endif

    // Pipe array: one pipe per child to collect timing result
    int pipes[n_procs][2];
    for (int i = 0; i < n_procs; ++i) {
        if (pipe(pipes[i]) != 0) {
            perror("pipe"); exit(1);
        }
    }

    // Wall-clock start (parent measures total time across all children)
    auto wall_start = std::chrono::high_resolution_clock::now();

    // Fork N children
    pid_t pids[n_procs];
    for (int i = 0; i < n_procs; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) { perror("fork"); exit(1); }

        if (pids[i] == 0) {
            // Child
            close(pipes[i][0]);
            double ms = 0.0;
            run_child_workload(n_kernels, BUF_N, &ms);
            write_result(pipes[i][1], ms);
            close(pipes[i][1]);
            exit(0);
        }
    }

    // Parent: close write ends, collect results
    double child_times[n_procs];
    memset(child_times, 0, sizeof(child_times));
    for (int i = 0; i < n_procs; ++i) {
        close(pipes[i][1]);
        ssize_t r = read(pipes[i][0], &child_times[i], sizeof(double));
        if (r != sizeof(double)) child_times[i] = -1.0;
        close(pipes[i][0]);
    }

    // Wait for all children
    for (int i = 0; i < n_procs; ++i) {
        int status;
        waitpid(pids[i], &status, 0);
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    // Report
    printf("=== MPS Benchmark Results ===\n");
    printf("  Wall time (all %d processes): %.2f ms\n", n_procs, wall_ms);
    printf("  Per-kernel average latency (each process):\n");
    double total_child = 0;
    for (int i = 0; i < n_procs; ++i) {
        double per_kernel_us = (child_times[i] / n_kernels) * 1e3;
        printf("    proc[%d]: total=%.2f ms, per-kernel=%.2f µs\n",
               i, child_times[i], per_kernel_us);
        total_child += child_times[i];
    }
    double avg_per_kernel = (total_child / n_procs / n_kernels) * 1e3;
    printf("  Average per-kernel latency: %.2f µs\n", avg_per_kernel);
    printf("\n");
    printf("Interpretation:\n");
    printf("  Without MPS: context switches cause serialization — wall time ≈\n"
           "  sum of per-process times.  Per-kernel latency is ~100 µs on V100.\n");
    printf("  With MPS: kernels overlap in hardware — wall time ≈ single-process\n"
           "  time.  Per-kernel latency drops to ~10 µs (driver path).\n");
    printf("\nTo enable MPS: sudo bash gpu_engine/mps/setup_mps.sh\n");
    printf("To disable:    echo quit | sudo nvidia-cuda-mps-control\n");
    printf("               sudo nvidia-smi -i 0 -c DEFAULT\n");

    return 0;
}
