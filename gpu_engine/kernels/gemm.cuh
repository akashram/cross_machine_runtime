#pragma once
// GEMM kernel variants: naive → shared-memory tiled → Tensor Core (WMMA).
//
// All variants compute C(M×N) = A(M×K) × B(K×N) with row-major layout.
//
// Variant 1 — naive (gemm_naive)
//   One thread per output element.  Inner loop over K.  No shared memory.
//   Bottleneck: every element of A and B is read from global memory K times.
//   Expected: ~5–20 TFLOPS on V100/A100 (heavily bandwidth-bound at large K).
//
// Variant 2 — shared-memory tiled (gemm_tiled<TILE>)
//   TILE×TILE blocks; each block loads one TILE×TILE tile of A and B into
//   __shared__ memory, then computes the partial dot-products.
//   Global memory reads reduced from 2*M*N*K to 2*M*N*K/TILE.
//   TILE=16: 256 threads, good for small-to-medium sizes.
//   TILE=32: 1024 threads, better for large sizes (stays within occupancy).
//
// Variant 3 — Tensor Core via WMMA (gemm_wmma)
//   Requires Volta+ (sm_70+).  A and B must be __half; C is float.
//   M must be multiple of 64, N and K multiples of 16.
//   Each warp computes a 16×16 output tile using nvcuda::wmma ops.
//   Expected: 60–130 TFLOPS on A100.
//
// Variant 4 — cuBLAS (in gemm_bench.cu)
//   Baseline reference; not defined here since it uses cublasHandle_t.

#include <cuda_runtime.h>
#include <mma.h>

// -----------------------------------------------------------------------
// Variant 1: naive
// -----------------------------------------------------------------------

__global__ void gemm_naive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// -----------------------------------------------------------------------
// Variant 2: shared-memory tiled
//
// Block = TILE×TILE threads, each thread computes one C[row][col].
// Each iteration of the tile loop:
//   - Load A[row, t*TILE : (t+1)*TILE] into As[ty][tx]
//   - Load B[t*TILE : (t+1)*TILE, col] into Bs[ty][tx]
//   - Accumulate dot product from shared memory
//
// Edge tiles (M/N/K not multiples of TILE) are handled by zero-padding.
// -----------------------------------------------------------------------

template<int TILE>
__global__ void gemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col]  : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// -----------------------------------------------------------------------
// Variant 3: Tensor Core via WMMA
//
// One warp per 16×16 output tile.  Block has WMMA_WARPS_PER_BLOCK warps
// arranged vertically: each covers a different set of 16 output rows.
//
// Constraints:
//   - A and B must be __half (FP16)
//   - M must be a multiple of (WMMA_WARPS_PER_BLOCK × 16)
//   - N and K must be multiples of 16
//
// For non-aligned sizes: pad A, B, C to the next valid multiple before
// calling this kernel.
//
// Launch dimensions:
//   dim3 block(WMMA_WARPS_PER_BLOCK * 32, 1)  ← 4 warps = 128 threads
//   dim3 grid((M + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M,
//             (N + 15) / 16)
// -----------------------------------------------------------------------

constexpr int WMMA_WARPS_PER_BLOCK = 4;   // warps per block
constexpr int WMMA_BLOCK_M         = WMMA_WARPS_PER_BLOCK * 16;  // 64 rows per block

__global__ void gemm_wmma(const __half* __restrict__ A,
                           const __half* __restrict__ B,
                           float*        __restrict__ C,
                           int M, int N, int K) {
    using namespace nvcuda;

    constexpr int WM = 16, WN = 16, WK = 16;  // WMMA tile dimensions

    int warp_id  = threadIdx.x / 32;
    int warp_row = blockIdx.x * WMMA_BLOCK_M + warp_id * WM;
    int warp_col = blockIdx.y * WN;

    // Entire warp skips if its tile is out of bounds
    if (warp_row >= M || warp_col >= N) return;

    wmma::fragment<wmma::matrix_a,    WM, WN, WK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WM, WN, WK, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float>                   acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WK) {
        // A tile: rows [warp_row, warp_row+16), cols [k, k+16)
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
        // B tile: rows [k, k+16), cols [warp_col, warp_col+16)
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, acc_frag, N,
                            wmma::mem_row_major);
}

// -----------------------------------------------------------------------
// Launch helpers — compute grid dimensions from matrix size
// -----------------------------------------------------------------------

inline dim3 gemm_naive_grid(int M, int N, int block = 16) {
    return dim3((N + block - 1) / block, (M + block - 1) / block);
}

template<int TILE>
inline dim3 gemm_tiled_grid(int M, int N) {
    return dim3((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
}

inline dim3 gemm_wmma_grid(int M, int N) {
    return dim3((M + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M, (N + 15) / 16);
}
