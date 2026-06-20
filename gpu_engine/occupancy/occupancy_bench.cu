// occupancy_bench.cu — sweep block sizes and measure occupancy per kernel
// TODO: run on GPU hardware

#include "occupancy.h"
#include <cstdio>

// Dummy kernels used as occupancy measurement targets
__global__ void dummy_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    // TODO: implement on GPU hardware
    // std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};
    // std::vector<size_t> shared_mems = {0, 4096, 8192, 16384, 32768, 49152};
    // OccupancyReport report;
    // sweep_occupancy((const void*)dummy_add, block_sizes, shared_mems, report);
    // print_report(report);
    printf("occupancy_bench: STUB — run on GPU hardware\n");
    return 0;
}
