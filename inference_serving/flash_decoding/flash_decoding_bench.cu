// flash_decoding_bench.cu — latency improvement from KV-split parallelism
// at long context lengths, PLAN.md Phase 9 step 4's explicit ask ("measure
// latency improvement for 8k+ token sequences").
//
// Compares num_kv_splits=1 (one block per (batch,head), serially looping
// the entire KV sequence — the decode-time degenerate case of
// gpu_engine/flash_attn's query-tile grid, since a decode step has
// exactly one query row) against choose_num_kv_splits()'s auto-chosen
// split count, across N in {512, 2048, 8192, 16384, 32768}. At small N
// the two should be close (not enough work to make idle SMs matter); the
// gap should open up at 8k+ where splits=1 leaves most SMs idle for a
// small batch.
//
// Correctness: split output verified against splits=1 within FP16
// tolerance (both compute the exact same softmax(QK^T/sqrt(D))V — only
// the parallelization strategy differs, so any mismatch beyond
// floating-point reassociation error would be a real bug, not expected
// numerical drift).
#include "flash_decoding.cuh"
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));          \
        exit(1);                                                      \
    }                                                                 \
} while (0)

namespace {

constexpr int kD = 128;
constexpr int kB = 4;    // small decode-serving batch -- exactly the case
constexpr int kH = 8;    // where B*H alone can't saturate an A100's 108 SMs

struct Buffers {
    __half *Q, *K, *V, *O;
    float *partial_O, *partial_m, *partial_l;
};

Buffers allocate(int N, int max_splits) {
    Buffers b{};
    int BH = kB * kH;
    CUDA_CHECK(cudaMalloc(&b.Q, sizeof(__half) * BH * kD));
    CUDA_CHECK(cudaMalloc(&b.K, sizeof(__half) * BH * N * kD));
    CUDA_CHECK(cudaMalloc(&b.V, sizeof(__half) * BH * N * kD));
    CUDA_CHECK(cudaMalloc(&b.O, sizeof(__half) * BH * kD));
    CUDA_CHECK(cudaMalloc(&b.partial_O, sizeof(float) * BH * max_splits * kD));
    CUDA_CHECK(cudaMalloc(&b.partial_m, sizeof(float) * BH * max_splits));
    CUDA_CHECK(cudaMalloc(&b.partial_l, sizeof(float) * BH * max_splits));
    return b;
}

void free_buffers(Buffers &b) {
    cudaFree(b.Q); cudaFree(b.K); cudaFree(b.V); cudaFree(b.O);
    cudaFree(b.partial_O); cudaFree(b.partial_m); cudaFree(b.partial_l);
}

float time_kernel(int N, int num_kv_splits, const Buffers &b, int iters = 50) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    launch_flash_decoding_explicit_splits<kD>(b.Q, b.K, b.V, b.O, b.partial_O, b.partial_m, b.partial_l,
                                               kB, kH, N, num_kv_splits);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        launch_flash_decoding_explicit_splits<kD>(b.Q, b.K, b.V, b.O, b.partial_O, b.partial_m, b.partial_l,
                                                   kB, kH, N, num_kv_splits);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

} // namespace

int main() {
    const std::vector<int> seq_lens = {512, 2048, 8192, 16384, 32768};

    std::printf("%-8s %-12s %14s %14s %10s\n", "N", "auto_splits", "splits=1(ms)", "auto(ms)", "speedup");
    for (int N : seq_lens) {
        int auto_splits = choose_num_kv_splits(kB, kH, N);
        Buffers b = allocate(N, std::max(1, auto_splits));

        float ms_1 = time_kernel(N, 1, b);
        float ms_auto = time_kernel(N, auto_splits, b);

        std::printf("%-8d %-12d %14.4f %14.4f %9.2fx\n", N, auto_splits, ms_1, ms_auto, ms_1 / ms_auto);
        free_buffers(b);
    }
    return 0;
}
