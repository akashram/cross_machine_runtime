// gemm_bench.cu — four-way GEMM comparison: naive / tiled / WMMA / cuBLAS
// TODO: run on GPU hardware

#include "gemm.cuh"
#include <cublas_v2.h>
#include <cstdio>

int main() {
    // TODO: implement on GPU hardware
    // Sizes to test: M=N=K in {512, 1024, 2048, 4096, 8192}
    // For each size:
    //   run gemm_naive, gemm_tiled<16>, gemm_tiled<32>, gemm_wmma, cublasSgemm
    //   measure TFLOPS = 2*M*N*K / elapsed_s / 1e12
    //   verify outputs match cuBLAS to within 1e-3
    //
    // Expected results (A100):
    //   naive:        ~1–5 TFLOPS
    //   tiled:        ~15–30 TFLOPS
    //   WMMA:         ~60–100 TFLOPS
    //   cuBLAS:       ~100–130 TFLOPS (FP32), ~312 TFLOPS (TF32)
    printf("gemm_bench: STUB — run on GPU hardware\n");
    return 0;
}
