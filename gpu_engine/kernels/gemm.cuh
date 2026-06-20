#pragma once
#include <mma.h>
// GEMM kernel variants — TODO: implement on GPU hardware

// Variant 1: naive — one thread per C[i][j]
__global__ void gemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K);

// Variant 2: shared-memory tiled — TILE_SIZE x TILE_SIZE blocks
template <int TILE>
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K);

// Variant 3: Tensor Core via WMMA (requires Volta+, FP16 A/B, FP32 C)
// M/N/K must be multiples of 16
__global__ void gemm_wmma(const __half* A, const __half* B, float* C,
                           int M, int N, int K);

// Variant 4: cuBLAS wrapper (for baseline comparison)
// Implemented in gemm_bench.cu using cublasSgemm
