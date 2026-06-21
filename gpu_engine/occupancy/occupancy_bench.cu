// occupancy_bench.cu — measure theoretical occupancy across block sizes and
// shared memory amounts for three kernels with different resource profiles.
//
// Kernels:
//   light_kernel   — minimal regs, no shared mem → typically hits hard block
//                    count limit before register/smem limits.
//   smem_kernel    — takes dynamic shared memory; demonstrates the smem cliff
//                    where one extra byte drops blocks/SM by 1.
//   regpressure_kernel — uses a longer compute chain to increase register count
//                    (compiler hint: volatile prevents elision). Shows how
//                    high register pressure reduces active warps.
//
// Expected output: README.md results table filled in from this run.

#include "gpu_engine/occupancy/occupancy.h"
#include <cstdio>
#include <vector>

// -----------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------

// Minimal register pressure — roughly 8–12 regs/thread on most architectures.
__global__ void light_kernel(const float* __restrict__ A,
                              float* __restrict__ B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) B[i] = A[i] + 1.0f;
}

// Shared memory consumer — caller passes dynamic smem via third kernel arg.
// Uses the smem as a scratch buffer (one float per thread).
__global__ void smem_kernel(const float* __restrict__ A,
                             float* __restrict__ B, int N) {
    extern __shared__ float scratch[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        scratch[threadIdx.x] = A[i];
        __syncthreads();
        B[i] = scratch[threadIdx.x] * 2.0f;
    }
}

// Higher register pressure: unroll a chain of FMAs the compiler can't fold.
// volatile forces the compiler to keep intermediate values live (→ more regs).
__global__ void regpressure_kernel(const float* __restrict__ A,
                                    float* __restrict__ B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    volatile float v = A[i];
    float acc = v;
    // 16 FMAs — enough to push register count up noticeably on most GPUs
    for (int j = 0; j < 16; ++j) acc = acc * 1.0001f + 0.0001f;
    B[i] = acc;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

#define CUDA_CHECK(call) do {                                       \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(_e));        \
        exit(1);                                                    \
    }                                                               \
} while (0)

int main() {
    // Print device name for README context
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d) — maxWarps/SM=%d  smem/SM=%zu KB\n\n",
           prop.name, prop.major, prop.minor,
           prop.maxThreadsPerMultiProcessor / 32,
           prop.sharedMemPerMultiprocessor / 1024);

    const std::vector<int>    block_sizes  = {32, 64, 128, 256, 512, 1024};
    const std::vector<size_t> smem_zero    = {0};
    // smem sweep: 0 B up to 48 KB in steps that let us see the block-count cliff
    const std::vector<size_t> smem_sweep   = {
        0, 4096, 8192, 12288, 16384, 24576, 32768, 49152
    };

    // ---- light_kernel ------------------------------------------------
    {
        int best_bs = 0, min_grid = 0;
        find_best_block_size((const void*)light_kernel, best_bs, min_grid);
        printf("light_kernel: cudaOccupancyMaxPotentialBlockSize → "
               "best_block_size=%d  min_grid_size=%d\n\n", best_bs, min_grid);

        OccupancyReport r;
        r.kernel_name = "light_kernel";
        sweep_occupancy((const void*)light_kernel, block_sizes, smem_zero, r);
        print_occupancy_report(r);
    }

    // ---- smem_kernel — sweep shared memory sizes ----------------------
    {
        int best_bs = 0, min_grid = 0;
        find_best_block_size((const void*)smem_kernel, best_bs, min_grid, /*smem=*/0);
        printf("\nsmem_kernel: cudaOccupancyMaxPotentialBlockSize → "
               "best_block_size=%d  min_grid_size=%d\n\n", best_bs, min_grid);

        // Sweep with fixed block_size=256, varying smem
        const std::vector<int> bs256 = {256};
        OccupancyReport r;
        r.kernel_name = "smem_kernel (block=256, smem sweep)";
        sweep_occupancy((const void*)smem_kernel, bs256, smem_sweep, r);
        print_occupancy_report(r);

        // Also sweep block sizes with smem = blockDim.x * sizeof(float)
        // (one float per thread — the actual usage in this kernel)
        printf("\nsmem_kernel: block_size sweep, smem = block_size × 4B\n");
        OccupancyReport r2;
        r2.kernel_name = "smem_kernel (proportional smem)";
        for (int bs : block_sizes) {
            if (bs <= 0 || bs % 32 != 0 || bs > 1024) continue;
            size_t smem = static_cast<size_t>(bs) * sizeof(float);
            try {
                auto res = measure_occupancy((const void*)smem_kernel, bs, smem);
                r2.sweep.push_back(res);
                if (res.theoretical_occupancy > r2.best.theoretical_occupancy)
                    r2.best = res;
            } catch (...) {}
        }
        print_occupancy_report(r2);
    }

    // ---- regpressure_kernel ------------------------------------------
    {
        int best_bs = 0, min_grid = 0;
        find_best_block_size((const void*)regpressure_kernel, best_bs, min_grid);
        printf("\nregpressure_kernel: cudaOccupancyMaxPotentialBlockSize → "
               "best_block_size=%d  min_grid_size=%d\n\n", best_bs, min_grid);

        OccupancyReport r;
        r.kernel_name = "regpressure_kernel";
        sweep_occupancy((const void*)regpressure_kernel, block_sizes, smem_zero, r);
        print_occupancy_report(r);
    }

    return 0;
}
