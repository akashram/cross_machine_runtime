#pragma once
// Flash Attention forward kernel — Dao et al. 2022 (https://arxiv.org/abs/2205.14135)
//
// Computes O = softmax(QK^T / sqrt(D)) V without ever materializing the full
// N×N score matrix in HBM.  The key insight is online softmax: we accumulate
// the output O tile-by-tile, maintaining a running max and sum that let us
// correct previous partial results as each new K/V tile is processed.
//
// Memory complexity: O(N) SRAM per block vs O(N²) HBM for naive attention.
//
// Thread-block structure
// ----------------------
// One block per (batch, head, query_tile).
//   blockDim.x  = Br   (one thread per query row in the tile)
//   gridDim.x   = B * H * ceil(N / Br)
//
// Each thread independently manages its query row: loads Q[q,:] into registers,
// maintains running (m_i, l_i, o_i[D]), and writes the final O[q,:] and L[q].
//
// Shared memory layout (all threads load K/V collaboratively):
//   float K_s[Bc][D]   key tile
//   float V_s[Bc][D]   value tile
// Total: 8 * Bc * D bytes.  For Bc=64, D=64: 32 KB.  For D=128 use Bc=32.
//
// Template parameters
// -------------------
//   Br  — query rows per block (choose Br = blockDim.x)
//   Bc  — key/value cols per tile (SRAM budget: 8*Bc*D ≤ shared_mem_per_SM)
//   D   — head dimension (must equal actual head_dim at launch time)
//
// Causal masking
// --------------
// When causal=true, S[q_global][kv_col] = -inf for kv_col > q_global.
// This is applied per-element so no extra memory is needed.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template <int Br, int Bc, int D>
__global__ void flash_attn_fwd(
    const __half* __restrict__ Q,   // [B*H, N, D] — batch and head dimensions flattened
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    float*        __restrict__ L,   // [B*H, N] log-sum-exp, saved for backward
    int N,                          // sequence length
    float scale,                    // 1 / sqrt(D)
    bool causal
) {
    // Decode grid index into (bh, q_tile)
    const int num_q_tiles = (N + Br - 1) / Br;
    const int bh     = blockIdx.x / num_q_tiles;
    const int q_tile = blockIdx.x % num_q_tiles;
    const int q_start = q_tile * Br;
    const int q_global = q_start + threadIdx.x;

    // Offset to this (batch, head) slice in the flattened [B*H, N, D] layout
    const long long off = (long long)bh * N * D;
    const __half* Qbh = Q + off;
    const __half* Kbh = K + off;
    const __half* Vbh = V + off;
    __half*       Obh = O + off;
    float*        Lbh = L + (long long)bh * N;

    const bool active = (q_global < N);

    // Load this thread's query row into registers.
    // Inactive threads keep zeros (they still participate in __syncthreads).
    float q_reg[D];
#pragma unroll
    for (int d = 0; d < D; ++d)
        q_reg[d] = active ? __half2float(Qbh[q_global * D + d]) : 0.0f;

    // Running online softmax state per query row
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[D];
#pragma unroll
    for (int d = 0; d < D; ++d) o_i[d] = 0.0f;

    // Shared memory: key and value tiles (loaded collaboratively, stored as float)
    __shared__ float K_s[Bc][D];
    __shared__ float V_s[Bc][D];

    const int num_kv_tiles = (N + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * Bc;

        // All threads load K tile and V tile from HBM to SRAM.
        // Even inactive threads participate so the barrier is valid.
        for (int i = threadIdx.x; i < Bc * D; i += Br) {
            int row = i / D, d = i % D;
            int kv_row = kv_start + row;
            K_s[row][d] = (kv_row < N) ? __half2float(Kbh[kv_row * D + d]) : 0.0f;
            V_s[row][d] = (kv_row < N) ? __half2float(Vbh[kv_row * D + d]) : 0.0f;
        }
        __syncthreads();

        if (active) {
            // Compute S[j] = Q[q_global] · K[kv_start+j] * scale, with causal masking
            float S[Bc];
#pragma unroll
            for (int j = 0; j < Bc; ++j) {
                int kv_col = kv_start + j;
                if (kv_col >= N || (causal && kv_col > q_global)) {
                    S[j] = -INFINITY;
                    continue;
                }
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < D; ++d) dot += q_reg[d] * K_s[j][d];
                S[j] = dot * scale;
            }

            // Online softmax: find block max
            float m_ij = S[0];
#pragma unroll
            for (int j = 1; j < Bc; ++j) m_ij = fmaxf(m_ij, S[j]);

            // Update running state
            float m_new   = fmaxf(m_i, m_ij);
            float rescale = (m_i > -INFINITY) ? expf(m_i - m_new) : 0.0f;

            l_i *= rescale;
#pragma unroll
            for (int d = 0; d < D; ++d) o_i[d] *= rescale;

            // Accumulate P_ij * V_j
#pragma unroll
            for (int j = 0; j < Bc; ++j) {
                float pij = (S[j] > -INFINITY) ? expf(S[j] - m_new) : 0.0f;
                l_i += pij;
#pragma unroll
                for (int d = 0; d < D; ++d) o_i[d] += pij * V_s[j][d];
            }

            m_i = m_new;
        }

        __syncthreads();  // before overwriting K_s/V_s in next iteration
    }

    // Normalise and write output
    if (active) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d)
            Obh[q_global * D + d] = __float2half(o_i[d] * inv_l);
        // L[q] = log-sum-exp = m_i + log(l_i)
        Lbh[q_global] = m_i + logf(fmaxf(l_i, 1e-6f));
    }
}

// -----------------------------------------------------------------------
// Launch helpers — select tile sizes based on D and available SRAM.
// -----------------------------------------------------------------------

// 8*Bc*D bytes of shared memory must fit in the SM's SRAM pool.
// With 48 KB limit: Bc = 48*1024 / (8*D).
// D=64  → Bc=96  (round down to 64 to stay power-of-two)
// D=128 → Bc=48  (round down to 32)
// Use Br=Bc for square tiles (balanced load between Q and K/V).

inline void launch_flash_attn_fwd_d64(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    int B, int H, int N,
    bool causal, cudaStream_t stream = 0)
{
    constexpr int Br = 64, Bc = 64, D = 64;
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    int BH = B * H;
    int num_q_tiles = (N + Br - 1) / Br;
    dim3 grid(BH * num_q_tiles), block(Br);
    flash_attn_fwd<Br, Bc, D><<<grid, block, 0, stream>>>(Q, K, V, O, L, N, scale, causal);
}

inline void launch_flash_attn_fwd_d128(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* L,
    int B, int H, int N,
    bool causal, cudaStream_t stream = 0)
{
    constexpr int Br = 64, Bc = 32, D = 128;
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    int BH = B * H;
    int num_q_tiles = (N + Br - 1) / Br;
    dim3 grid(BH * num_q_tiles), block(Br);
    flash_attn_fwd<Br, Bc, D><<<grid, block, 0, stream>>>(Q, K, V, O, L, N, scale, causal);
}
