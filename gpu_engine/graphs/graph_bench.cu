// graph_bench.cu — measure CUDA graph replay vs eager launch overhead
// TODO: run on GPU hardware
#include "graph_runner.h"
#include <cstdio>

int main() {
    // TODO: implement on GPU hardware
    // 1. Build a chain of N simple kernels (e.g., add_kernel)
    // 2. Launch eagerly N times, measure CPU time with RDTSC
    // 3. Capture as graph, replay N times, measure CPU time
    // 4. Print eager vs graph overhead per launch
    printf("graph_bench: STUB — run on GPU hardware\n");
    return 0;
}
