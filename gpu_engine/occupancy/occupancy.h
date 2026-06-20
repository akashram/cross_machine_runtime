#pragma once
#include <vector>

// TODO: implement on CUDA GPU hardware

struct OccupancyResult {
    int block_size;
    size_t shared_mem_bytes;
    int registers_per_thread;
    int active_blocks_per_sm;
    double theoretical_occupancy;  // 0.0 – 1.0
};

struct OccupancyReport {
    std::string kernel_name;
    std::vector<OccupancyResult> sweep;
    OccupancyResult best;  // highest theoretical_occupancy
};

// Query CUDA occupancy for a single (kernel, block_size, shared_mem) combination.
OccupancyResult measure_occupancy(const void* kernel,
                                   int block_size,
                                   size_t shared_mem_bytes);

// Sweep block sizes and shared_mem values; populate report.
void sweep_occupancy(const void* kernel,
                     const std::vector<int>& block_sizes,
                     const std::vector<size_t>& shared_mem_bytes,
                     OccupancyReport& report);
