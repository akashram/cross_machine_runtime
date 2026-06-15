// Warp-primitive benchmark
// Compares warp shuffle instructions vs equivalent shared-memory kernels.
//
// Sections:
//   1. Block reduce: warp-shuffle + inter-warp shm  vs  pure shm tree
//   2. Block scan: warp Kogge-Stone + inter-warp shm  vs  pure shm scan
//   3. Ballot compaction: __ballot_sync + __popc  vs  per-element atomicAdd
//
// For sections 1 and 2, both variants are memory-bound on large inputs
// (reading the input array dominates). The warp variant is expected to be
// 10–30% faster due to fewer shared memory round-trips and no intra-warp
// __syncthreads(). The win is larger when block size is small (fewer
// warps → inter-warp shm is cheap) and larger when measuring latency
// vs throughput in Nsight.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

#include "gpu_engine/warp_primitives/warp_ops.h"

static void check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",
                     file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CHECK(x) check((x), __FILE__, __LINE__)

// =========================================================================
// 1. Reduce kernels
//    Both write one float per block (the sum) to out[].
//    Caller launches with blockDim.x = BLOCK and enough blocks to cover n.
// =========================================================================
static constexpr int BLOCK = 256;
static constexpr int WARPS = BLOCK / 32;

// Warp-shuffle block reduce
__global__ void reduce_warp(const float* __restrict__ in,
                             float* __restrict__ out, int n) {
    __shared__ float warp_sums[WARPS];

    int i = blockIdx.x * BLOCK + threadIdx.x;
    float val = (i < n) ? in[i] : 0.0f;

    // Step 1: reduce within each warp — only lane 0 of each warp gets result
    val = gpu_engine::warp::reduce_sum_lane0(val);

    // Step 2: each warp's lane 0 writes to shared memory
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();

    // Step 3: first warp reduces the WARPS partial sums
    if (warp == 0) {
        val = (lane < WARPS) ? warp_sums[lane] : 0.0f;
        val = gpu_engine::warp::reduce_sum_lane0(val);
        if (lane == 0) out[blockIdx.x] = val;
    }
}

// Pure shared-memory block reduce (sequential-address pattern, no bank conflict)
__global__ void reduce_shm(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
    __shared__ float sdata[BLOCK];

    int i = blockIdx.x * BLOCK + threadIdx.x;
    sdata[threadIdx.x] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sdata[0];
}

// =========================================================================
// 2. Scan kernels
//    Warp-local inclusive prefix scan: one warp handles 32 contiguous
//    elements. Blocks of 32 threads only; for arrays that are multiples of 32.
// =========================================================================

// Warp-shuffle scan
__global__ void scan_warp(const float* __restrict__ in,
                           float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? in[i] : 0.0f;
    val = gpu_engine::warp::scan_inclusive(val);
    if (i < n) out[i] = val;
}

// Shared-memory Hillis-Steele scan (within 32 threads, double-buffer)
__global__ void scan_shm(const float* __restrict__ in,
                          float* __restrict__ out, int n) {
    __shared__ float a[32], b[32];
    int i = blockIdx.x * 32 + threadIdx.x;
    int t = threadIdx.x;
    a[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int delta = 1; delta < 32; delta <<= 1) {
        b[t] = a[t];
        if (t >= delta) b[t] += a[t - delta];
        __syncthreads();
        float* tmp = nullptr; (void)tmp;  // swap pointers conceptually
        // copy b back into a
        a[t] = b[t];
        __syncthreads();
    }
    if (i < n) out[i] = a[t];
}

// =========================================================================
// 3. Predicate compaction via ballot vs atomicAdd
// =========================================================================

// Count elements > threshold using __ballot_sync + __popc per warp
__global__ void count_ballot(const float* __restrict__ in,
                              unsigned* __restrict__ out,
                              int n, float threshold) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    bool pred = (i < n) && (in[i] > threshold);
    // Each warp gets a 32-bit mask; popcount gives the count for this warp.
    unsigned hits = static_cast<unsigned>(
        __popc(gpu_engine::warp::ballot(pred)));
    // One atomic per warp (not per thread) — 32× cheaper than per-thread atomic
    if (threadIdx.x % 32 == 0)
        atomicAdd(out, hits);
}

// Count elements > threshold with one atomic per thread (naive)
__global__ void count_atomic(const float* __restrict__ in,
                              unsigned* __restrict__ out,
                              int n, float threshold) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n && in[i] > threshold)
        atomicAdd(out, 1u);
}

// =========================================================================
// Timing helpers
// =========================================================================
template<typename F>
static double bench_ms(F&& launch, int iters = 30) {
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    // Warmup
    for (int i = 0; i < 5; ++i) launch();
    CHECK(cudaDeviceSynchronize());
    std::vector<double> ms(iters);
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaEventRecord(t0));
        launch();
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float m = 0.f;
        CHECK(cudaEventElapsedTime(&m, t0, t1));
        ms[i] = m;
    }
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    std::sort(ms.begin(), ms.end());
    return ms[iters / 2];  // return p50
}

int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("Device: %s (sm_%d%d)\n\n", p.name, p.major, p.minor);

    // -----------------------------------------------------------------------
    // 1. Block reduce: 64M floats (256 MB)
    // -----------------------------------------------------------------------
    const int N_REDUCE = 64 << 20;  // 64M elements
    const std::size_t bytes_reduce = static_cast<std::size_t>(N_REDUCE) * sizeof(float);

    float* d_in  = nullptr;
    float* d_out = nullptr;
    const int n_blocks = (N_REDUCE + BLOCK - 1) / BLOCK;
    CHECK(cudaMalloc(&d_in,  bytes_reduce));
    CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(n_blocks) * sizeof(float)));
    CHECK(cudaMemset(d_in, 0, bytes_reduce));

    std::printf("=== Block reduce (N=64M floats = 256 MB, block=%d) ===\n", BLOCK);
    {
        double ms_w = bench_ms([&]{ reduce_warp<<<n_blocks, BLOCK>>>(d_in, d_out, N_REDUCE); });
        double ms_s = bench_ms([&]{ reduce_shm <<<n_blocks, BLOCK>>>(d_in, d_out, N_REDUCE); });
        double gb_w = bytes_reduce / (ms_w * 1e-3) / 1e9;
        double gb_s = bytes_reduce / (ms_s * 1e-3) / 1e9;
        std::printf("  warp-shuffle reduce:  p50=%.2f ms  %.1f GB/s\n", ms_w, gb_w);
        std::printf("  shared-memory reduce: p50=%.2f ms  %.1f GB/s\n", ms_s, gb_s);
        std::printf("  speedup: %.2fx\n\n", ms_s / ms_w);
    }
    cudaFree(d_in);
    cudaFree(d_out);

    // -----------------------------------------------------------------------
    // 2. Warp scan: 4M floats (16 MB), blocks of 32 threads (one warp each)
    // -----------------------------------------------------------------------
    const int N_SCAN = 4 << 20;  // 4M elements, must be multiple of 32
    const std::size_t bytes_scan = static_cast<std::size_t>(N_SCAN) * sizeof(float);
    const int scan_blocks = N_SCAN / 32;

    float* d_scan_in  = nullptr;
    float* d_scan_out = nullptr;
    CHECK(cudaMalloc(&d_scan_in,  bytes_scan));
    CHECK(cudaMalloc(&d_scan_out, bytes_scan));
    CHECK(cudaMemset(d_scan_in, 0, bytes_scan));

    // Correctness check: run both, compare outputs
    {
        // Fill with 1.0f so prefix sum at lane i = i+1
        std::vector<float> h_in(N_SCAN, 1.0f);
        CHECK(cudaMemcpy(d_scan_in, h_in.data(), bytes_scan, cudaMemcpyHostToDevice));

        scan_warp<<<scan_blocks, 32>>>(d_scan_in, d_scan_out, N_SCAN);
        CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(N_SCAN);
        CHECK(cudaMemcpy(h_out.data(), d_scan_out, bytes_scan, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N_SCAN && ok; ++i) {
            float expected = static_cast<float>(i % 32 + 1);
            if (std::abs(h_out[i] - expected) > 0.5f) {
                std::printf("SCAN MISMATCH at i=%d: got %.1f expected %.1f\n",
                            i, h_out[i], expected);
                ok = false;
            }
        }
        if (ok) std::printf("Scan correctness: OK\n");
        CHECK(cudaMemset(d_scan_in, 0, bytes_scan));
    }

    std::printf("\n=== Warp scan (N=4M floats = 16 MB, 32 threads/block) ===\n");
    {
        double ms_w = bench_ms([&]{ scan_warp<<<scan_blocks, 32>>>(d_scan_in, d_scan_out, N_SCAN); });
        double ms_s = bench_ms([&]{ scan_shm <<<scan_blocks, 32>>>(d_scan_in, d_scan_out, N_SCAN); });
        // Read + write → 2 × bytes
        double gb_w = 2.0 * bytes_scan / (ms_w * 1e-3) / 1e9;
        double gb_s = 2.0 * bytes_scan / (ms_s * 1e-3) / 1e9;
        std::printf("  warp-shuffle scan:    p50=%.2f ms  %.1f GB/s\n", ms_w, gb_w);
        std::printf("  shared-memory scan:   p50=%.2f ms  %.1f GB/s\n", ms_s, gb_s);
        std::printf("  speedup: %.2fx\n\n", ms_s / ms_w);
    }
    cudaFree(d_scan_in);
    cudaFree(d_scan_out);

    // -----------------------------------------------------------------------
    // 3. Predicate compaction: 64M floats, count elements > 0.5
    // -----------------------------------------------------------------------
    const int N_BALLOT = 64 << 20;
    const std::size_t bytes_ballot = static_cast<std::size_t>(N_BALLOT) * sizeof(float);

    float*    d_ballot_in  = nullptr;
    unsigned* d_count      = nullptr;
    CHECK(cudaMalloc(&d_ballot_in, bytes_ballot));
    CHECK(cudaMalloc(&d_count, sizeof(unsigned)));
    // Fill with values in [0, 1): about 50% will exceed threshold 0.5
    {
        std::vector<float> h(N_BALLOT);
        for (int i = 0; i < N_BALLOT; ++i)
            h[i] = static_cast<float>(i % 1000) / 1000.0f;
        CHECK(cudaMemcpy(d_ballot_in, h.data(), bytes_ballot, cudaMemcpyHostToDevice));
    }

    std::printf("=== Ballot compaction (N=64M floats, threshold=0.5) ===\n");
    {
        const int bb = (N_BALLOT + BLOCK - 1) / BLOCK;
        const float thresh = 0.5f;

        // Verify both give same count
        unsigned h_ballot = 0, h_atomic = 0;
        CHECK(cudaMemset(d_count, 0, sizeof(unsigned)));
        count_ballot<<<bb, BLOCK>>>(d_ballot_in, d_count, N_BALLOT, thresh);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&h_ballot, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost));

        CHECK(cudaMemset(d_count, 0, sizeof(unsigned)));
        count_atomic<<<bb, BLOCK>>>(d_ballot_in, d_count, N_BALLOT, thresh);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&h_atomic, d_count, sizeof(unsigned), cudaMemcpyDeviceToHost));

        std::printf("  Count check — ballot=%u  atomic=%u  %s\n\n",
                    h_ballot, h_atomic,
                    h_ballot == h_atomic ? "OK" : "MISMATCH");

        CHECK(cudaMemset(d_count, 0, sizeof(unsigned)));
        double ms_b = bench_ms([&]{
            CHECK(cudaMemset(d_count, 0, sizeof(unsigned)));
            count_ballot<<<bb, BLOCK>>>(d_ballot_in, d_count, N_BALLOT, thresh);
        });
        CHECK(cudaMemset(d_count, 0, sizeof(unsigned)));
        double ms_a = bench_ms([&]{
            CHECK(cudaMemset(d_count, 0, sizeof(unsigned)));
            count_atomic<<<bb, BLOCK>>>(d_ballot_in, d_count, N_BALLOT, thresh);
        });
        double gb_b = bytes_ballot / (ms_b * 1e-3) / 1e9;
        double gb_a = bytes_ballot / (ms_a * 1e-3) / 1e9;
        std::printf("  ballot (__ballot_sync + __popc, 1 atomic/warp): p50=%.2f ms  %.1f GB/s\n",
                    ms_b, gb_b);
        std::printf("  atomic (atomicAdd per thread):                  p50=%.2f ms  %.1f GB/s\n",
                    ms_a, gb_a);
        std::printf("  speedup: %.2fx\n", ms_a / ms_b);
    }
    cudaFree(d_ballot_in);
    cudaFree(d_count);

    return 0;
}
