// Shared memory primitives benchmark
// =========================================================================
//
// Three sections:
//
// 1. Matrix transpose — the canonical bank conflict demonstration.
//    Three variants on an N×N matrix:
//      a) Naive:   no shared memory, uncoalesced global writes
//      b) Tiled:   32×32 tile in shared memory — coalesced I/O but 32-way
//                  bank conflicts when reading columns out of the tile
//      c) Padded:  32×33 tile — same coalesced I/O, zero bank conflicts
//
//    Bank conflict analysis (TILE = 32):
//      Tiled:  bank of tile[r][c] = (r*32 + c) % 32 = c % 32
//              All 32 rows reading column c hit bank c → 32-way conflict.
//      Padded: bank of tile[r][c] = (r*33 + c) % 32 = (r + c) % 32
//              Reading column c, row r: bank = (r + c) % 32 → all 32 rows
//              hit different banks → no conflict.
//
// 2. Block-level inclusive scan — uses shared_ops.h (warp scan + inter-warp
//    shared memory) vs a naive sequential shared-memory scan for comparison.
//    Correctness: all-1 input → result at index i within block = i + 1.
//
// 3. Explicit bank conflict micro-benchmark — isolates the serialization
//    cost of N-way bank conflicts without the confounding of global memory.
//    Launches one warp, 10000 iterations of shared memory reads.
//    stride-1: each of 32 threads hits a different bank  → no conflict
//    stride-32: all 32 threads hit bank 0                → 32-way conflict

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

#include "gpu_engine/shared_mem/shared_ops.h"

static void check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",
                     file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CHECK(x) check((x), __FILE__, __LINE__)

template<typename F>
static double bench_ms(F&& launch, int iters = 30) {
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
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
    return ms[iters / 2];
}

// =========================================================================
// 1. Transpose kernels  (N × N float matrix)
// =========================================================================
static constexpr int TILE = 32;

// a) Naive: no shared memory. Reads are coalesced; writes are strided by N.
__global__ void transpose_naive(const float* __restrict__ in,
                                 float* __restrict__ out, int N) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];
}

// b) Tiled with bank conflicts: 32×32 tile, 32-way conflict on column read.
__global__ void transpose_tiled(const float* __restrict__ in,
                                 float* __restrict__ out, int N) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    // Transposed write: read tile column threadIdx.x (all rows) → 32-way conflict
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

// c) Padded tile: +1 column eliminates bank conflicts.
__global__ void transpose_padded(const float* __restrict__ in,
                                  float* __restrict__ out, int N) {
    __shared__ float tile[TILE][TILE + 1];  // +1 column: shifts each row's bank mapping

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

// =========================================================================
// 2. Block-level scan kernels
// =========================================================================
static constexpr int SCAN_BLOCK = 256;
static constexpr int SCAN_WARPS = SCAN_BLOCK / 32;

// Using shared_ops.h block scan
__global__ void scan_block_ops(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    __shared__ float scratch[SCAN_WARPS];
    int i = blockIdx.x * SCAN_BLOCK + threadIdx.x;
    float val = (i < n) ? in[i] : 0.0f;
    val = gpu_engine::block::block_scan_inclusive(val, scratch);
    if (i < n) out[i] = val;
}

// Naive shared-memory Hillis-Steele scan (within block)
__global__ void scan_block_naive(const float* __restrict__ in,
                                  float* __restrict__ out, int n) {
    __shared__ float a[SCAN_BLOCK];
    int i = blockIdx.x * SCAN_BLOCK + threadIdx.x;
    int t = threadIdx.x;
    a[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // log2(SCAN_BLOCK) = 8 passes, each with a __syncthreads()
    for (int delta = 1; delta < SCAN_BLOCK; delta <<= 1) {
        float tmp = (t >= delta) ? a[t - delta] : 0.0f;
        __syncthreads();
        a[t] += tmp;
        __syncthreads();
    }
    if (i < n) out[i] = a[t];
}

// =========================================================================
// 3. Bank conflict micro-benchmark
//    One warp per block, read shared memory in a loop.
//    stride=1:  no conflict   (32 threads × 32 banks = 1:1 mapping)
//    stride=32: 32-way conflict (all 32 threads hit bank 0)
// =========================================================================
template<int STRIDE>
__global__ void shm_stride_bench(float* out, int iters) {
    // Shared array large enough for stride-32 access: 32 threads × stride 32
    __shared__ float shm[32 * 32];
    // Initialize so the compiler can't elide the reads
    shm[threadIdx.x] = static_cast<float>(threadIdx.x + 1);
    __syncwarp();

    float acc = 0.0f;
    const int idx = threadIdx.x * STRIDE;  // fixed index per thread per iteration
    for (int i = 0; i < iters; ++i)
        acc += shm[idx];

    if (threadIdx.x == 0) atomicAdd(out, acc);
}

// =========================================================================
// main
// =========================================================================
int main() {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    double hbm_peak = static_cast<double>(p.memoryClockRate) * 1e3 * 2.0
                      * (p.memoryBusWidth / 8.0) / 1e9;
    std::printf("Theoretical HBM bandwidth: %.1f GB/s\n\n", hbm_peak);

    // -----------------------------------------------------------------------
    // 1. Transpose (4096 × 4096 float = 64 MB)
    // -----------------------------------------------------------------------
    const int N = 4096;
    const std::size_t mat_bytes = static_cast<std::size_t>(N) * N * sizeof(float);
    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in,  mat_bytes));
    CHECK(cudaMalloc(&d_out, mat_bytes));
    CHECK(cudaMemset(d_in, 0, mat_bytes));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Verify padded transpose is correct
    {
        std::vector<float> h_in(N * N), h_out(N * N);
        for (int i = 0; i < N * N; ++i) h_in[i] = static_cast<float>(i);
        CHECK(cudaMemcpy(d_in, h_in.data(), mat_bytes, cudaMemcpyHostToDevice));
        transpose_padded<<<grid, block>>>(d_in, d_out, N);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_out.data(), d_out, mat_bytes, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int r = 0; r < N && ok; ++r)
            for (int c = 0; c < N && ok; ++c)
                if (std::fabs(h_out[r * N + c] - h_in[c * N + r]) > 0.5f) {
                    std::printf("Transpose MISMATCH at (%d,%d)\n", r, c);
                    ok = false;
                }
        if (ok) std::printf("Transpose correctness: OK\n\n");
        CHECK(cudaMemset(d_in, 0, mat_bytes));
    }

    // GB/s = 2 * mat_bytes (read + write) / time
    auto bw = [&](double ms) { return 2.0 * mat_bytes / (ms * 1e-3) / 1e9; };

    std::printf("=== Matrix transpose (%d×%d, %.0f MB) ===\n",
                N, N, mat_bytes / 1e6);
    std::printf("%-28s  %8s  %10s\n", "Variant", "p50 ms", "GB/s");
    {
        double ms = bench_ms([&]{ transpose_naive <<<grid, block>>>(d_in, d_out, N); });
        std::printf("%-28s  %8.2f  %10.1f\n", "naive (uncoalesced writes)", ms, bw(ms));
    }
    {
        double ms = bench_ms([&]{ transpose_tiled <<<grid, block>>>(d_in, d_out, N); });
        std::printf("%-28s  %8.2f  %10.1f  (32-way bank conflict)\n", "tiled 32×32", ms, bw(ms));
    }
    {
        double ms = bench_ms([&]{ transpose_padded<<<grid, block>>>(d_in, d_out, N); });
        std::printf("%-28s  %8.2f  %10.1f  (no conflict)\n", "padded 32×33", ms, bw(ms));
    }
    std::printf("Theoretical peak: %.1f GB/s\n\n", hbm_peak);
    cudaFree(d_in); cudaFree(d_out);

    // -----------------------------------------------------------------------
    // 2. Block scan (16M floats = 64 MB)
    // -----------------------------------------------------------------------
    const int N_SCAN = 16 << 20;
    const std::size_t scan_bytes = static_cast<std::size_t>(N_SCAN) * sizeof(float);
    float *d_scan_in = nullptr, *d_scan_out = nullptr;
    CHECK(cudaMalloc(&d_scan_in,  scan_bytes));
    CHECK(cudaMalloc(&d_scan_out, scan_bytes));

    // Correctness check: input = all 1s → output[i within block] = i + 1
    {
        std::vector<float> h_in(N_SCAN, 1.0f);
        CHECK(cudaMemcpy(d_scan_in, h_in.data(), scan_bytes, cudaMemcpyHostToDevice));

        int n_blocks = (N_SCAN + SCAN_BLOCK - 1) / SCAN_BLOCK;
        scan_block_ops<<<n_blocks, SCAN_BLOCK>>>(d_scan_in, d_scan_out, N_SCAN);
        CHECK(cudaDeviceSynchronize());

        std::vector<float> h_out(N_SCAN);
        CHECK(cudaMemcpy(h_out.data(), d_scan_out, scan_bytes, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N_SCAN && ok; ++i) {
            float expected = static_cast<float>(i % SCAN_BLOCK + 1);
            if (std::fabs(h_out[i] - expected) > 0.5f) {
                std::printf("Scan MISMATCH at %d: got %.1f expected %.1f\n",
                            i, h_out[i], expected);
                ok = false;
            }
        }
        if (ok) std::printf("Block scan correctness: OK\n\n");
        CHECK(cudaMemset(d_scan_in, 0, scan_bytes));
    }

    auto scan_bw = [&](double ms) {
        // read input + write output = 2 × bytes
        return 2.0 * scan_bytes / (ms * 1e-3) / 1e9;
    };

    std::printf("=== Block inclusive scan (N=16M floats, block=%d) ===\n", SCAN_BLOCK);
    std::printf("%-36s  %8s  %10s\n", "Variant", "p50 ms", "GB/s");
    {
        int nb = (N_SCAN + SCAN_BLOCK - 1) / SCAN_BLOCK;
        double ms = bench_ms([&]{
            scan_block_ops<<<nb, SCAN_BLOCK>>>(d_scan_in, d_scan_out, N_SCAN);
        });
        std::printf("%-36s  %8.2f  %10.1f  (warp scan + inter-warp shm)\n",
                    "block::scan_inclusive (shared_ops.h)", ms, scan_bw(ms));
    }
    {
        int nb = (N_SCAN + SCAN_BLOCK - 1) / SCAN_BLOCK;
        double ms = bench_ms([&]{
            scan_block_naive<<<nb, SCAN_BLOCK>>>(d_scan_in, d_scan_out, N_SCAN);
        });
        std::printf("%-36s  %8.2f  %10.1f  (Hillis-Steele, log2(N) barriers)\n",
                    "naive Hillis-Steele shm", ms, scan_bw(ms));
    }
    std::printf("\n");
    cudaFree(d_scan_in); cudaFree(d_scan_out);

    // -----------------------------------------------------------------------
    // 3. Bank conflict micro-benchmark
    //    32 threads (one warp), 10000 iterations of shm reads.
    //    stride-1:  thread k reads shm[k]    — bank k    — no conflict
    //    stride-32: thread k reads shm[k*32] — bank 0    — 32-way conflict
    // -----------------------------------------------------------------------
    std::printf("=== Bank conflict micro-benchmark (32 threads, 10000 iters) ===\n");
    float* d_dummy = nullptr;
    CHECK(cudaMalloc(&d_dummy, sizeof(float)));
    CHECK(cudaMemset(d_dummy, 0, sizeof(float)));

    constexpr int ITERS = 10000;
    {
        double ms1  = bench_ms([&]{ shm_stride_bench<1> <<<1, 32>>>(d_dummy, ITERS); });
        double ms32 = bench_ms([&]{ shm_stride_bench<32><<<1, 32>>>(d_dummy, ITERS); });
        std::printf("  stride-1  (no conflict):   p50=%.3f ms\n", ms1);
        std::printf("  stride-32 (32-way conflict): p50=%.3f ms\n", ms32);
        std::printf("  slowdown from bank conflict: %.1fx\n", ms32 / ms1);
        std::printf("  (ideal 32-way serialization → 32× slower)\n");
    }
    cudaFree(d_dummy);

    return 0;
}
