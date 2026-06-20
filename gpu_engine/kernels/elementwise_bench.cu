// elementwise_bench.cu — benchmark add/mul/relu/gelu/softmax throughput
// TODO: run on GPU hardware

#include "elementwise.cuh"
#include <cstdio>

int main() {
    // TODO: implement on GPU hardware
    // for each kernel:
    //   allocate device buffers
    //   warm up 10 iterations
    //   time 100 iterations with cudaEvent
    //   compute GB/s = bytes_transferred / elapsed_s
    //   run coalescing_check.sh and record ratio
    printf("elementwise_bench: STUB — run on GPU hardware\n");
    return 0;
}
