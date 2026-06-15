// Stream manager benchmark
// Measures:
//   1. cudaStreamCreate / cudaStreamDestroy latency vs StreamPool::acquire
//   2. Sequential H2D + kernel vs overlapped H2D ‖ kernel
//
// The overlap experiment is the key deliverable: it demonstrates that a
// GPU with separate DMA engines and compute SMs can run an H2D transfer
// on one stream concurrently with a compute kernel on another stream.
// Total time should collapse from (T_copy + T_kernel) to max(T_copy, T_kernel).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

#include "gpu_engine/streams/stream_pool.h"
#include "gpu_engine/streams/event_pool.h"

static void check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",
                     file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CHECK(x) check((x), __FILE__, __LINE__)

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Busy-sleep kernel — each thread spins for `cycles` SM clock ticks.
// Launched with enough blocks to fill all SMs so the GPU is genuinely busy
// and cannot be preempted onto a different workload.
__global__ void busy_sleep(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

// Measure GPU-side elapsed time (ms) for a no-op lambda using cudaEvents.
// The lambda receives a cudaStream_t to submit work to.
template<typename F>
static double gpu_ms(cudaStream_t stream, F&& f) {
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0, stream));
    f(stream);
    CHECK(cudaEventRecord(t1, stream));
    CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return static_cast<double>(ms);
}

static void print_latency(const char* label, std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    std::printf("  %-36s  p50=%6.2f µs  p99=%6.2f µs  mean=%6.2f µs\n",
                label,
                v[v.size() / 2],
                v[v.size() * 99 / 100],
                mean);
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("Device: %s (sm_%d%d, %d SMs, clockRate %d kHz)\n\n",
                p.name, p.major, p.minor,
                p.multiProcessorCount, p.clockRate);

    const int N = 200;  // iterations for latency measurements

    // -----------------------------------------------------------------------
    // 1. Stream / event creation latency
    // -----------------------------------------------------------------------
    std::printf("=== Creation latency (N=%d) ===\n", N);

    {
        std::vector<double> create_us(N), destroy_us(N);
        for (int i = 0; i < N; ++i) {
            cudaStream_t s;
            double t0 = now_us();
            cudaStreamCreate(&s);
            double t1 = now_us();
            cudaStreamDestroy(s);
            double t2 = now_us();
            create_us[i]  = t1 - t0;
            destroy_us[i] = t2 - t1;
        }
        print_latency("cudaStreamCreate", create_us);
        print_latency("cudaStreamDestroy", destroy_us);
    }

    {
        std::vector<double> create_us(N), destroy_us(N);
        for (int i = 0; i < N; ++i) {
            cudaEvent_t e;
            double t0 = now_us();
            cudaEventCreate(&e);
            double t1 = now_us();
            cudaEventDestroy(e);
            double t2 = now_us();
            create_us[i]  = t1 - t0;
            destroy_us[i] = t2 - t1;
        }
        print_latency("cudaEventCreate", create_us);
        print_latency("cudaEventDestroy", destroy_us);
    }

    {
        gpu_engine::StreamPool pool(16);
        std::vector<double> acq_us(N);
        for (int i = 0; i < N; ++i) {
            double t0 = now_us();
            auto g = pool.acquire();
            double t1 = now_us();
            acq_us[i] = t1 - t0;
            (void)g;
        }
        print_latency("StreamPool::acquire (no contention)", acq_us);
    }

    std::printf("\n");

    // -----------------------------------------------------------------------
    // 2. Calibrate busy_sleep kernel
    //
    // clock64() counts SM clock cycles. clockRate is in kHz.
    // cycles_for_ms(ms) = ms * clockRate * 1000 / 1000 = ms * clockRate.
    // We calibrate against wall time to account for boost clocks.
    // -----------------------------------------------------------------------
    const double target_ms = 15.0;
    long long sleep_cycles = static_cast<long long>(target_ms * p.clockRate);

    // Calibration: run kernel, measure actual GPU time, rescale once.
    {
        cudaStream_t cal_stream;
        CHECK(cudaStreamCreateWithFlags(&cal_stream, cudaStreamNonBlocking));

        double actual_ms = gpu_ms(cal_stream, [&](cudaStream_t s) {
            busy_sleep<<<p.multiProcessorCount, 128, 0, s>>>(sleep_cycles);
        });

        // Rescale to hit target_ms
        sleep_cycles = static_cast<long long>(
            static_cast<double>(sleep_cycles) * (target_ms / actual_ms));

        // Verify
        double verified_ms = gpu_ms(cal_stream, [&](cudaStream_t s) {
            busy_sleep<<<p.multiProcessorCount, 128, 0, s>>>(sleep_cycles);
        });
        std::printf("Sleep kernel calibration: target=%.1f ms, actual=%.2f ms\n",
                    target_ms, verified_ms);

        cudaStreamDestroy(cal_stream);
    }

    // Transfer size — pick something that takes a similar duration to the kernel.
    // ~15 ms on a T4 (PCIe 3.0 x16 ≈ 12 GB/s) → 256 MB × (1/12) s/GB ≈ 21 ms.
    // We'll use 128 MB to stay close to kernel time.
    const std::size_t kBytes = 128ull << 20;  // 128 MB

    void* host_src = nullptr;
    void* dev_dst  = nullptr;
    CHECK(cudaMallocHost(&host_src, kBytes));
    CHECK(cudaMalloc(&dev_dst, kBytes));
    std::memset(host_src, 0xab, kBytes);

    // -----------------------------------------------------------------------
    // 3. Individual operation timings (for reference)
    // -----------------------------------------------------------------------
    std::printf("\n=== Individual operation timings ===\n");

    double copy_ms, kernel_ms;
    {
        cudaStream_t s;
        CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

        copy_ms = gpu_ms(s, [&](cudaStream_t st) {
            CHECK(cudaMemcpyAsync(dev_dst, host_src, kBytes,
                                  cudaMemcpyHostToDevice, st));
        });
        std::printf("  H2D transfer  (128 MB, pinned): %.2f ms\n", copy_ms);

        kernel_ms = gpu_ms(s, [&](cudaStream_t st) {
            busy_sleep<<<p.multiProcessorCount, 128, 0, st>>>(sleep_cycles);
        });
        std::printf("  busy_sleep kernel (calibrated): %.2f ms\n", kernel_ms);
        std::printf("  Sequential sum (expected):      %.2f ms\n",
                    copy_ms + kernel_ms);
        std::printf("  Overlap ceiling (expected):     %.2f ms\n",
                    std::max(copy_ms, kernel_ms));

        cudaStreamDestroy(s);
    }

    // -----------------------------------------------------------------------
    // 4. Sequential vs overlapped — wall time over ITERS repetitions
    // -----------------------------------------------------------------------
    std::printf("\n=== Sequential vs overlapped (ITERS=20, wall time) ===\n");
    const int ITERS = 20;

    // Sequential: copy then kernel, single stream
    {
        gpu_engine::StreamPool pool(1);
        std::vector<double> wall_ms(ITERS);
        for (int i = 0; i < ITERS; ++i) {
            auto s = pool.acquire();
            double t0 = now_us();
            CHECK(cudaMemcpyAsync(dev_dst, host_src, kBytes,
                                  cudaMemcpyHostToDevice, s));
            busy_sleep<<<p.multiProcessorCount, 128, 0, s>>>(sleep_cycles);
            CHECK(cudaStreamSynchronize(s));
            wall_ms[i] = (now_us() - t0) / 1000.0;
        }
        std::sort(wall_ms.begin(), wall_ms.end());
        double mean = std::accumulate(wall_ms.begin(), wall_ms.end(), 0.0) / ITERS;
        std::printf("  Sequential  — mean=%.2f ms  p50=%.2f ms  p99=%.2f ms\n",
                    mean, wall_ms[ITERS/2], wall_ms[ITERS*99/100]);
    }

    // Overlapped: copy on stream 0, kernel on stream 1
    {
        gpu_engine::StreamPool pool(2);
        std::vector<double> wall_ms(ITERS);
        for (int i = 0; i < ITERS; ++i) {
            auto s0 = pool.acquire();
            auto s1 = pool.acquire();
            double t0 = now_us();
            // Submit both without waiting — GPU runs them concurrently.
            CHECK(cudaMemcpyAsync(dev_dst, host_src, kBytes,
                                  cudaMemcpyHostToDevice, s0));
            busy_sleep<<<p.multiProcessorCount, 128, 0, s1>>>(sleep_cycles);
            CHECK(cudaStreamSynchronize(s0));
            CHECK(cudaStreamSynchronize(s1));
            wall_ms[i] = (now_us() - t0) / 1000.0;
        }
        std::sort(wall_ms.begin(), wall_ms.end());
        double mean = std::accumulate(wall_ms.begin(), wall_ms.end(), 0.0) / ITERS;
        std::printf("  Overlapped  — mean=%.2f ms  p50=%.2f ms  p99=%.2f ms\n",
                    mean, wall_ms[ITERS/2], wall_ms[ITERS*99/100]);
        std::printf("\n  Speedup (mean): %.2fx  (ideal: %.2fx)\n",
                    (copy_ms + kernel_ms) / mean,
                    (copy_ms + kernel_ms) / std::max(copy_ms, kernel_ms));
    }

    cudaFreeHost(host_src);
    cudaFree(dev_dst);
    return 0;
}
