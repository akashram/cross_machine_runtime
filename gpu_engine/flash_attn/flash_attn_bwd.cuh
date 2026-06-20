#pragma once
// Flash Attention backward kernel — TODO: implement on GPU hardware
// Recomputes attention weights from saved L (log-sum-exp) to avoid storing S.
//
// Inputs: Q, K, V, O, dO, L (from forward)
// Outputs: dQ, dK, dV

template <int Br, int Bc, int D>
__global__ void flash_attn_bwd(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ O,
    const __half* __restrict__ dO,  // upstream gradient
    const float*  __restrict__ L,   // log-sum-exp from forward
    __half*       __restrict__ dQ,
    __half*       __restrict__ dK,
    __half*       __restrict__ dV,
    int B, int H, int N,
    float scale,
    bool causal
);
