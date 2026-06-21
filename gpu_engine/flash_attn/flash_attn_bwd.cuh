#pragma once
// Flash Attention backward kernel — Dao et al. 2022 Algorithm 2
//
// Recomputes attention scores from saved L (log-sum-exp) rather than reading
// the full N×N S matrix from HBM.  HBM I/O is O(N·D) instead of O(N²).
//
// Saved from forward: O [B*H, N, D], L [B*H, N]
// Inputs:            Q, K, V, dO (upstream gradient)
// Outputs:           dQ, dK, dV
//
// Algorithm (outer = Q tile, inner = K/V tile)
// --------------------------------------------
// For query tile i:
//   Load Q_i, O_i, dO_i, L_i from HBM.
//   D_i  = rowsum(dO_i ⊙ O_i)   (Br values; "delta" for softmax gradient)
//   For each K/V tile j:
//     Load K_j, V_j from HBM.
//     Recompute S_ij = Q_i K_j^T * scale  (Br × Bc)
//     Recompute P_ij = exp(S_ij - L_i)    (Br × Bc)
//     dV_j   += P_ij^T dO_i               atomic: many Q tiles → same K/V tile
//     dP_ij   = dO_i V_j^T                (Br × Bc)
//     dS_ij   = P_ij ⊙ (dP_ij − D_i)     softmax Jacobian
//     dQ_i   += dS_ij K_j * scale         (Br × D) — no atomic, Q tile is unique
//     dK_j   += dS_ij^T Q_i * scale       atomic: many Q tiles → same K/V tile
//
// Atomics: dK and dV require atomicAdd since multiple query blocks write to the
// same K/V positions.  This is correct but may be slow for small D.  The
// FlashAttention-2 paper restructures the loop to avoid atomics for dK/dV;
// that optimisation is left as a TODO in the README.
//
// Thread-block structure (same as forward)
// ----------------------------------------
// blockDim.x = Br, one thread per query row in the tile.
// gridDim.x  = B * H * ceil(N / Br).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template <int Br, int Bc, int D>
__global__ void flash_attn_bwd(
    const __half* __restrict__ Q,    // [B*H, N, D]
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ O,    // forward output
    const __half* __restrict__ dO,   // upstream gradient
    const float*  __restrict__ L,    // [B*H, N] log-sum-exp from forward
    __half*       __restrict__ dQ,   // output gradient for Q
    __half*       __restrict__ dK,   // output gradient for K (atomically accumulated)
    __half*       __restrict__ dV,   // output gradient for V (atomically accumulated)
    int N,
    float scale,
    bool causal
) {
    const int num_q_tiles = (N + Br - 1) / Br;
    const int bh      = blockIdx.x / num_q_tiles;
    const int q_tile  = blockIdx.x % num_q_tiles;
    const int q_start = q_tile * Br;
    const int q_global = q_start + threadIdx.x;

    const long long off = (long long)bh * N * D;
    const __half* Qbh  = Q  + off;
    const __half* Kbh  = K  + off;
    const __half* Vbh  = V  + off;
    const __half* Obh  = O  + off;
    const __half* dObh = dO + off;
    __half*       dQbh = dQ + off;
    __half*       dKbh = dK + off;
    __half*       dVbh = dV + off;
    const float*  Lbh  = L  + (long long)bh * N;

    const bool active = (q_global < N);

    // Load Q[q_global], O[q_global], dO[q_global] into registers.
    float q_reg[D], o_reg[D], do_reg[D];
#pragma unroll
    for (int d = 0; d < D; ++d) {
        q_reg[d]  = active ? __half2float(Qbh [q_global * D + d]) : 0.0f;
        o_reg[d]  = active ? __half2float(Obh [q_global * D + d]) : 0.0f;
        do_reg[d] = active ? __half2float(dObh[q_global * D + d]) : 0.0f;
    }

    // D_i = sum_d(dO[q,d] * O[q,d])  — scalar per query row
    float D_i = 0.0f;
    if (active) {
#pragma unroll
        for (int d = 0; d < D; ++d) D_i += do_reg[d] * o_reg[d];
    }

    // Running dQ accumulator (no atomics needed — only this block writes dQ_i)
    float dq_acc[D];
#pragma unroll
    for (int d = 0; d < D; ++d) dq_acc[d] = 0.0f;

    // Log-sum-exp for this query row (saved during forward)
    float L_i = active ? Lbh[q_global] : 0.0f;

    // Shared memory: key, value, and their gradient accumulators
    __shared__ float K_s[Bc][D];
    __shared__ float V_s[Bc][D];

    const int num_kv_tiles = (N + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * Bc;

        // Load K_j and V_j into shared memory (all threads collaborate)
        for (int i = threadIdx.x; i < Bc * D; i += Br) {
            int row = i / D, d = i % D;
            int kv_row = kv_start + row;
            K_s[row][d] = (kv_row < N) ? __half2float(Kbh[kv_row * D + d]) : 0.0f;
            V_s[row][d] = (kv_row < N) ? __half2float(Vbh[kv_row * D + d]) : 0.0f;
        }
        __syncthreads();

        if (active) {
            // Recompute attention scores S[j] = Q[q_global] · K[kv_start+j] * scale
            float S[Bc], P[Bc];
#pragma unroll
            for (int j = 0; j < Bc; ++j) {
                int kv_col = kv_start + j;
                if (kv_col >= N || (causal && kv_col > q_global)) {
                    S[j] = -INFINITY;
                    P[j] = 0.0f;
                    continue;
                }
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < D; ++d) dot += q_reg[d] * K_s[j][d];
                S[j] = dot * scale;
                P[j] = expf(S[j] - L_i);  // recomputed probability
            }

            // dV_j += P_ij^T · dO_i
            // Thread qi contributes P[j] * do_reg[d] to dV[kv_start+j, d]
            // Atomic because other q_tiles also write to the same dV positions.
            for (int j = 0; j < Bc; ++j) {
                int kv_col = kv_start + j;
                if (kv_col >= N || P[j] == 0.0f) continue;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    float contrib = P[j] * do_reg[d];
                    // atomicAdd for __half not available on all architectures;
                    // use float-based atomic and write __half result.
                    // For Volta+: use atomicAdd directly on __half2 pairs.
                    // For simplicity: use float accumulation via a temporary.
                    // TODO on hardware: profile if this is the bottleneck and
                    // switch to FlashAttention-2 KV-outer loop to avoid atomics.
                    float* dst = reinterpret_cast<float*>(dVbh) + kv_col * D + d;
                    // dVbh is __half* — we can't use float* alias directly.
                    // Instead accumulate into __half via atomic on the correct type:
                    // On sm_70+ atomicAdd(__half*, __half) is supported.
#if __CUDA_ARCH__ >= 700
                    atomicAdd(dVbh + kv_col * D + d, __float2half(contrib));
#else
                    // On older hardware, use CAS loop on __half
                    // (rare: FlashAttention requires Volta+ for WMMA anyway)
                    unsigned short* addr = reinterpret_cast<unsigned short*>(dVbh + kv_col * D + d);
                    unsigned short old_val = *addr;
                    unsigned short assumed;
                    do {
                        assumed = old_val;
                        float updated = __half2float(__ushort_as_half(assumed)) + contrib;
                        old_val = atomicCAS(addr, assumed, __half_as_ushort(__float2half(updated)));
                    } while (assumed != old_val);
#endif
                }
            }

            // dP_ij[j] = sum_d(dO[d] * V_j[d])
            // dS_ij[j] = P[j] * (dP_ij[j] - D_i)   (softmax Jacobian)
            float dS[Bc];
#pragma unroll
            for (int j = 0; j < Bc; ++j) {
                float dP_j = 0.0f;
#pragma unroll
                for (int d = 0; d < D; ++d) dP_j += do_reg[d] * V_s[j][d];
                dS[j] = P[j] * (dP_j - D_i);
            }

            // dQ_i += dS_ij · K_j * scale  (no atomic: only this block writes dQ)
#pragma unroll
            for (int j = 0; j < Bc; ++j) {
                if (dS[j] == 0.0f) continue;
                float ds_scaled = dS[j] * scale;
#pragma unroll
                for (int d = 0; d < D; ++d) dq_acc[d] += ds_scaled * K_s[j][d];
            }

            // dK_j += dS_ij^T · Q_i * scale  (atomic: multiple q_tiles write same K pos)
#pragma unroll
            for (int j = 0; j < Bc; ++j) {
                int kv_col = kv_start + j;
                if (kv_col >= N || dS[j] == 0.0f) continue;
                float ds_scaled = dS[j] * scale;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    float contrib = ds_scaled * q_reg[d];
#if __CUDA_ARCH__ >= 700
                    atomicAdd(dKbh + kv_col * D + d, __float2half(contrib));
#else
                    unsigned short* addr = reinterpret_cast<unsigned short*>(dKbh + kv_col * D + d);
                    unsigned short old_val = *addr;
                    unsigned short assumed;
                    do {
                        assumed = old_val;
                        float updated = __half2float(__ushort_as_half(assumed)) + contrib;
                        old_val = atomicCAS(addr, assumed, __half_as_ushort(__float2half(updated)));
                    } while (assumed != old_val);
#endif
                }
            }
        }

        __syncthreads();
    }

    // Write accumulated dQ (no atomic needed)
    if (active) {
#pragma unroll
        for (int d = 0; d < D; ++d)
            dQbh[q_global * D + d] = __float2half(dq_acc[d]);
    }
}

// Launch helpers (matching forward's tile sizes)
inline void launch_flash_attn_bwd_d64(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* L,
    __half* dQ, __half* dK, __half* dV,
    int B, int H, int N, bool causal, cudaStream_t stream = 0)
{
    constexpr int Br = 64, Bc = 64, D = 64;
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    int num_q_tiles = (N + Br - 1) / Br;
    dim3 grid(B * H * num_q_tiles), block(Br);
    flash_attn_bwd<Br, Bc, D><<<grid, block, 0, stream>>>(
        Q, K, V, O, dO, L, dQ, dK, dV, N, scale, causal);
}

inline void launch_flash_attn_bwd_d128(
    const __half* Q, const __half* K, const __half* V,
    const __half* O, const __half* dO, const float* L,
    __half* dQ, __half* dK, __half* dV,
    int B, int H, int N, bool causal, cudaStream_t stream = 0)
{
    constexpr int Br = 64, Bc = 32, D = 128;
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    int num_q_tiles = (N + Br - 1) / Br;
    dim3 grid(B * H * num_q_tiles), block(Br);
    flash_attn_bwd<Br, Bc, D><<<grid, block, 0, stream>>>(
        Q, K, V, O, dO, L, dQ, dK, dV, N, scale, causal);
}
