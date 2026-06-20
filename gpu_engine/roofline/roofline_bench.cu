// roofline_bench.cu — measure peak FLOPS and HBM bandwidth, classify all kernels
// TODO: run on GPU hardware
#include "roofline.h"
#include <cstdio>

int main() {
    // TODO: implement on GPU hardware
    // 1. measure_peak_flops_tflops() — run 30s of cuBLAS SGEMM, take max
    // 2. measure_peak_bandwidth_gbs() — STREAM TRIAD kernel
    // 3. For each kernel (elementwise, gemm variants, flash_attn):
    //    - measure achieved_tflops and flops_per_byte
    //    - call classify_kernel and print roofline analysis
    printf("roofline_bench: STUB — run on GPU hardware\n");
    return 0;
}
