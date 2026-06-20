// hopper_bench.cu — benchmark TMA bandwidth and WGMMA vs WMMA throughput
// TODO: run on H100 (p5.48xlarge)
#include "tma.cuh"
#include "wgmma.cuh"
#include <cstdio>

int main() {
#if __CUDA_ARCH__ >= 900
    // TODO: implement on H100
    // 1. TMA bandwidth: measure memcpy_async+TMA vs cudaMemcpyAsync for 1MB/64MB/1GB
    // 2. WGMMA throughput: TFLOPS for M=N=K=4096 vs WMMA and cuBLAS
#endif
    printf("hopper_bench: STUB — run on H100 (p5.48xlarge)\n");
    return 0;
}
