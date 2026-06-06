#pragma once

// Cache-aware blocked matrix multiply: C = A * B  (all row-major)
// ====================================================================
//
// PROBLEM WITH NAIVE MATMUL
// --------------------------
// For an M×K × K×N multiply with large matrices, the naive (i,k,j) loop
// touches 3 × M × K × N / (cache_line / 4) cache lines total.  The B
// matrix is read K times (once per row of A), so its footprint is K × N
// floats streamed repeatedly.  When K × N > L2 size, every pass re-reads
// B from L3 or DRAM.
//
// TILING IDEA
// -----------
// Partition A into TM×TK tiles and B into TK×TN tiles.  For each pair
// of tiles the micro-kernel reads only 3T² floats (A, B, C tiles).
// Choose T so 3T²×4 ≤ L1 (32 KB) → T ≤ 52; practical sweet spot T=32.
//
// Working set by tier (square tiles, T floats per side):
//   T=16:  3072 B  ≈  3 KB  — deep in L1
//   T=32: 12288 B  ≈ 12 KB  — fits L1 (32 KB)
//   T=48: 27648 B  ≈ 27 KB  — fits L1
//   T=64: 49152 B  ≈ 48 KB  — fits L2 (256 KB)
//  T=128: 196608 B ≈192 KB  — fits L2
//  T=256: 786432 B ≈768 KB  — fits L3
//  T=512: 3145728 B ≈ 3 MB  — L3 / DRAM (same as untiled 512×512)
//
// LOOP ORDER
// ----------
// Outer (tile): i0 → k0 → j0
//   For fixed (i0,k0): A tile A[i0:,k0:] stays in L1 while we sweep j0.
//   For fixed j0: B tile B[k0:,j0:] is loaded once and reused TM times.
//   C tile C[i0:,j0:] accumulates across k0 passes.
//
// Inner (micro-kernel): i → k → j (j is vectorized)
//   a_ik = A[i][k] is a scalar register broadcast across the j strip.
//   B[k][j0..j0+TN] and C[i][j0..j0+TN] are sequential → auto-vectorized.

#include <algorithm>
#include <cstddef>

namespace cpu_engine::tiling {

// Returns the size in bytes of the three tiles (A, B, C) for a square tile T.
// Use this to annotate benchmark output with the cache tier.
[[nodiscard]] constexpr std::size_t tile_working_set_bytes(int T) noexcept {
    return static_cast<std::size_t>(3) * static_cast<std::size_t>(T) *
           static_cast<std::size_t>(T) * sizeof(float);
}

// ---- matmul_naive_f32 -------------------------------------------------------
// Plain (i,k,j) triple loop.  j is innermost and vectorized.
// Correct reference; also serves as the "no tiling" baseline in benchmarks.
//
// Precondition: C is pre-zeroed by the caller.
inline void matmul_naive_f32(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int K, int N) noexcept {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a_ik = A[i * K + k];
#pragma clang loop vectorize(enable)
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a_ik * B[k * N + j];
        }
    }
}

// ---- matmul_tiled_f32 -------------------------------------------------------
// Blocked multiply with runtime square tile size `tile`.
// tile controls how many rows/columns each block covers for all three matrices.
//
// Precondition: C is pre-zeroed by the caller.
inline void matmul_tiled_f32(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int K, int N,
                              int tile) noexcept {
    for (int i0 = 0; i0 < M; i0 += tile) {
        const int iend = std::min(i0 + tile, M);
        for (int k0 = 0; k0 < K; k0 += tile) {
            const int kend = std::min(k0 + tile, K);
            for (int j0 = 0; j0 < N; j0 += tile) {
                const int jend = std::min(j0 + tile, N);
                for (int i = i0; i < iend; ++i) {
                    for (int k = k0; k < kend; ++k) {
                        const float a_ik = A[i * K + k];
#pragma clang loop vectorize(enable)
                        for (int j = j0; j < jend; ++j)
                            C[i * N + j] += a_ik * B[k * N + j];
                    }
                }
            }
        }
    }
}

} // namespace cpu_engine::tiling
