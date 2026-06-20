#pragma once
// Flash Attention forward kernel — TODO: implement on GPU hardware
// Reference: Dao et al. 2022 (https://arxiv.org/abs/2205.14135)
//
// Algorithm sketch:
//   for each block of rows Br of Q:
//     init O_i = 0, l_i = 0, m_i = -inf
//     for each block of cols Bc of K, V:
//       load K_j, V_j from HBM to SRAM
//       S_ij = Q_i * K_j^T / sqrt(d)          # (Br x Bc) in SRAM
//       if causal: mask S_ij[i,j] = -inf for j > i + block_offset
//       m_ij = rowmax(S_ij)
//       P_ij = exp(S_ij - m_ij)               # online: subtract running max
//       l_ij = rowsum(P_ij)
//       update m_i, l_i with new block statistics
//       O_i = diag(exp(m_prev - m_new)) * O_i + P_ij * V_j
//     O_i /= l_i                              # normalize
//     write O_i, l_i, m_i to HBM

// Template params: Br=row block, Bc=col block, D=head dim
template <int Br, int Bc, int D>
__global__ void flash_attn_fwd(
    const __half* __restrict__ Q,   // [B, H, N, D]
    const __half* __restrict__ K,   // [B, H, N, D]
    const __half* __restrict__ V,   // [B, H, N, D]
    __half*       __restrict__ O,   // [B, H, N, D]
    float*        __restrict__ L,   // [B, H, N] — log-sum-exp for backward
    int B, int H, int N,
    float scale,
    bool causal
);
