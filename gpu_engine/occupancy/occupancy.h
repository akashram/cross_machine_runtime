#pragma once
// Occupancy tuner — wraps cudaOccupancyMaxActiveBlocksPerMultiprocessor and
// cudaOccupancyMaxPotentialBlockSize with a sweep API and a print helper.
//
// Theoretical occupancy = (active_blocks × block_warps) / max_warps_per_SM
//
// The three limiters:
//   1. Registers: each thread gets a slice of the SM register file (65536 on Ampere).
//      More regs/thread → fewer threads active → lower occupancy.
//   2. Shared memory: each block's smem comes out of the SM's shared memory pool
//      (typically 48–100 KB on Ampere). More smem/block → fewer blocks → lower occupancy.
//   3. Block count limit: SMs have a hard cap on concurrent blocks (typically 32 on Ampere).
//
// How to use the sweep results:
//   - Start with best_block_size from find_best_block_size().
//   - If a kernel uses shared memory, sweep smem sizes to find the cliff where
//     one additional block worth of smem drops active_blocks_per_sm by 1.
//   - Register pressure: visible in r.registers_per_thread after cudaFuncGetAttributes.
//     If high, try __launch_bounds__(block_size, min_blocks_per_sm) on the kernel.

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

struct OccupancyResult {
    int    block_size;
    size_t shared_mem_bytes;
    int    registers_per_thread;   // from cudaFuncGetAttributes
    int    active_blocks_per_sm;
    double theoretical_occupancy;  // 0.0–1.0
};

struct OccupancyReport {
    std::string kernel_name;
    std::vector<OccupancyResult> sweep;
    OccupancyResult best;          // entry with highest theoretical_occupancy
};

namespace detail {

#define OCC_CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                              \
    if (_e != cudaSuccess)                                                \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
} while (0)

inline int max_warps_per_sm() {
    cudaDeviceProp prop;
    OCC_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.maxThreadsPerMultiProcessor / 32;
}

} // namespace detail

inline OccupancyResult measure_occupancy(const void* kernel,
                                          int block_size,
                                          size_t shared_mem_bytes) {
    using namespace detail;
    cudaFuncAttributes attrs{};
    OCC_CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));

    int active_blocks = 0;
    OCC_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, block_size, shared_mem_bytes));

    int mwps = max_warps_per_sm();
    int block_warps = block_size / 32;

    OccupancyResult r;
    r.block_size            = block_size;
    r.shared_mem_bytes      = shared_mem_bytes;
    r.registers_per_thread  = attrs.numRegs;
    r.active_blocks_per_sm  = active_blocks;
    r.theoretical_occupancy = (mwps > 0)
        ? static_cast<double>(active_blocks * block_warps) / mwps
        : 0.0;
    return r;
}

// Sweep all (block_size, shared_mem) combinations and populate report.
// Invalid combinations (e.g. smem > SM capacity) are silently skipped.
inline void sweep_occupancy(const void* kernel,
                              const std::vector<int>& block_sizes,
                              const std::vector<size_t>& shared_mem_bytes,
                              OccupancyReport& report) {
    report.sweep.clear();
    report.best = {};
    report.best.theoretical_occupancy = -1.0;

    for (int bs : block_sizes) {
        if (bs <= 0 || bs % 32 != 0 || bs > 1024) continue;
        for (size_t smem : shared_mem_bytes) {
            try {
                auto r = measure_occupancy(kernel, bs, smem);
                report.sweep.push_back(r);
                if (r.theoretical_occupancy > report.best.theoretical_occupancy)
                    report.best = r;
            } catch (...) {}  // skip combinations the driver rejects
        }
    }
}

// Find the single block size that CUDA recommends to maximise occupancy
// (no shared memory constraint). Wraps cudaOccupancyMaxPotentialBlockSize.
inline void find_best_block_size(const void* kernel,
                                  int& best_block_size,
                                  int& min_grid_size,
                                  size_t dynamic_smem = 0) {
    detail::OCC_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &best_block_size, kernel, dynamic_smem, /*blockSizeLimit=*/0));
}

inline void print_occupancy_report(const OccupancyReport& report) {
    printf("\n=== Occupancy sweep: %s ===\n", report.kernel_name.c_str());
    printf("%-10s  %-12s  %-12s  %-10s  %s\n",
           "blk_size", "smem (B)", "regs/thd", "blks/SM", "occupancy");
    printf("%s\n", std::string(62, '-').c_str());

    for (const auto& r : report.sweep) {
        printf("%-10d  %-12zu  %-12d  %-10d  %.1f%%\n",
               r.block_size, r.shared_mem_bytes, r.registers_per_thread,
               r.active_blocks_per_sm, r.theoretical_occupancy * 100.0);
    }

    const auto& b = report.best;
    printf("\nBest: block_size=%-4d  smem=%-8zu  regs=%-3d  blks/SM=%-3d  occ=%.1f%%\n",
           b.block_size, b.shared_mem_bytes, b.registers_per_thread,
           b.active_blocks_per_sm, b.theoretical_occupancy * 100.0);
}
