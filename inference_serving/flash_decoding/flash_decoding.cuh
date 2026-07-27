#pragma once
// FlashDecoding — parallelizes single-query (decode-step) attention across
// the KV/sequence dimension, split-K-style. (FlashDecoding++, Hong et al.
// 2023; builds on gpu_engine/flash_attn's Dao et al. 2022 forward kernel.)
//
// The problem this solves: gpu_engine/flash_attn's kernel parallelizes
// over QUERY tiles (grid = B*H*ceil(N/Br)) — during autoregressive decode
// there is exactly one query row per (batch, head), so that grid collapses
// to B*H blocks total. On a long context (8k+ KV tokens), each of those
// B*H blocks then serially loops over every KV tile alone — with a small
// batch size (the common decode-serving case, one request at a time or a
// handful), B*H is nowhere near enough blocks to saturate the SMs, so
// most of the GPU sits idle while a few blocks grind through thousands of
// KV tiles sequentially. Splitting the KV dimension across additional
// blocks (this file) restores parallelism at exactly the point flash_attn's
// query-tile grid runs out of it — the two techniques aren't
// alternatives, they're the same online-softmax algorithm parallelized
// over the axis that actually has work to split at decode time.
//
// Two kernels, same online-softmax math as flash_attn_fwd.cuh's inner
// loop, split across a combine step:
//   1. flash_decoding_partial: grid = (B*H, num_kv_splits) — each block
//      handles ONE (batch,head) query row against ONE contiguous KV
//      chunk, producing a partial (unnormalized) output plus the running
//      max/sum needed to correctly combine it with every other split
//      later (identical per-chunk math to flash_attn_fwd's kv_tile loop
//      body, just writing intermediate state out instead of continuing
//      to the next chunk in the same block).
//   2. flash_decoding_combine: grid = (B*H) — merges num_kv_splits partial
//      results per query using the same online-softmax merge rule used
//      to combine two partial (m, l, o) states in flash_attn_fwd's
//      running-update step, generalized to N-way instead of pairwise.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <algorithm>

// ---------------------------------------------------------------------
// Kernel 1: per-split partial attention for one decode query.
// ---------------------------------------------------------------------
// Q: [B*H, D] — single query row per (batch,head), this IS decode.
// K, V: [B*H, N, D]
// partial_O: [B*H, num_kv_splits, D]
// partial_m, partial_l: [B*H, num_kv_splits]
template <int D, int kThreadsPerBlock>
__global__ void flash_decoding_partial(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float*        __restrict__ partial_O,
    float*        __restrict__ partial_m,
    float*        __restrict__ partial_l,
    int N,                  // total KV sequence length
    int num_kv_splits,
    float scale
) {
    const int bh    = blockIdx.x;
    const int split = blockIdx.y;
    const int tid   = threadIdx.x;

    const int chunk = (N + num_kv_splits - 1) / num_kv_splits;
    const int kv_start = split * chunk;
    const int kv_end   = min(kv_start + chunk, N);
    if (kv_start >= kv_end) {
        // This split has no work (N not evenly divisible by num_kv_splits
        // at the tail) — still write neutral state so combine's sum over
        // all splits stays correct without a branch there.
        if (tid == 0) {
            partial_m[bh * num_kv_splits + split] = -INFINITY;
            partial_l[bh * num_kv_splits + split] = 0.0f;
        }
        for (int d = tid; d < D; d += kThreadsPerBlock)
            partial_O[(bh * num_kv_splits + split) * D + d] = 0.0f;
        return;
    }

    const __half* Qbh = Q + (long long)bh * D;
    const __half* Kbh = K + (long long)bh * N * D;
    const __half* Vbh = V + (long long)bh * N * D;

    __shared__ float q_s[D];
    for (int d = tid; d < D; d += kThreadsPerBlock) q_s[d] = __half2float(Qbh[d]);
    __syncthreads();

    // Each thread accumulates a private partial state over a strided
    // subset of this split's KV rows, then a block-wide tree reduction
    // (via shared memory) combines them — same online-softmax merge rule
    // as the cross-split combine kernel below, just done once more at
    // thread granularity within a block.
    float m_local = -INFINITY, l_local = 0.0f;
    float o_local[D];
#pragma unroll
    for (int d = 0; d < D; ++d) o_local[d] = 0.0f;

    for (int kv = kv_start + tid; kv < kv_end; kv += kThreadsPerBlock) {
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) dot += q_s[d] * __half2float(Kbh[kv * D + d]);
        float s = dot * scale;

        float m_new = fmaxf(m_local, s);
        float rescale = (m_local > -INFINITY) ? expf(m_local - m_new) : 0.0f;
        float p = expf(s - m_new);

        l_local = l_local * rescale + p;
#pragma unroll
        for (int d = 0; d < D; ++d)
            o_local[d] = o_local[d] * rescale + p * __half2float(Vbh[kv * D + d]);
        m_local = m_new;
    }

    // Block-wide reduction of (m_local, l_local, o_local) across threads
    // into a single per-split result. Shared-memory tree reduction,
    // pairwise online-softmax merge at each step.
    __shared__ float m_s[kThreadsPerBlock];
    __shared__ float l_s[kThreadsPerBlock];
    __shared__ float o_s[kThreadsPerBlock][D];
    m_s[tid] = m_local;
    l_s[tid] = l_local;
#pragma unroll
    for (int d = 0; d < D; ++d) o_s[tid][d] = o_local[d];
    __syncthreads();

    for (int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m_a = m_s[tid], m_b = m_s[tid + stride];
            float m_new = fmaxf(m_a, m_b);
            float ra = (m_a > -INFINITY) ? expf(m_a - m_new) : 0.0f;
            float rb = (m_b > -INFINITY) ? expf(m_b - m_new) : 0.0f;
            l_s[tid] = l_s[tid] * ra + l_s[tid + stride] * rb;
#pragma unroll
            for (int d = 0; d < D; ++d)
                o_s[tid][d] = o_s[tid][d] * ra + o_s[tid + stride][d] * rb;
            m_s[tid] = m_new;
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_m[bh * num_kv_splits + split] = m_s[0];
        partial_l[bh * num_kv_splits + split] = l_s[0];
    }
    for (int d = tid; d < D; d += kThreadsPerBlock)
        partial_O[(bh * num_kv_splits + split) * D + d] = o_s[0][d];
}

// ---------------------------------------------------------------------
// Kernel 2: combine num_kv_splits partial results per query into the
// final normalized output. One block per (batch,head).
// ---------------------------------------------------------------------
template <int D>
__global__ void flash_decoding_combine(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __half*      __restrict__ O,
    int num_kv_splits
) {
    const int bh = blockIdx.x;

    float m = -INFINITY;
    for (int s = 0; s < num_kv_splits; ++s)
        m = fmaxf(m, partial_m[bh * num_kv_splits + s]);

    float l = 0.0f;
    float o[D];
#pragma unroll
    for (int d = 0; d < D; ++d) o[d] = 0.0f;

    for (int s = 0; s < num_kv_splits; ++s) {
        float ms = partial_m[bh * num_kv_splits + s];
        if (ms == -INFINITY) continue;  // this split had no KV rows
        float scale_s = expf(ms - m);
        l += partial_l[bh * num_kv_splits + s] * scale_s;
#pragma unroll
        for (int d = 0; d < D; ++d)
            o[d] += partial_O[(bh * num_kv_splits + s) * D + d] * scale_s;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
    for (int d = 0; d < D; ++d)
        O[bh * D + d] = __float2half(o[d] * inv_l);
}

// ---------------------------------------------------------------------
// Launch helper. num_kv_splits chosen so B*H*num_kv_splits gives the SM
// count enough blocks to saturate — the whole point of this technique.
// A100: 108 SMs; pick splits so B*H*splits is a small multiple of that.
// ---------------------------------------------------------------------
inline int choose_num_kv_splits(int B, int H, int N, int sm_count = 108) {
    int bh = B * H;
    if (bh >= sm_count) return 1;  // already enough parallelism from batch*heads alone
    int target_splits = (sm_count + bh - 1) / bh;
    // Don't split more finely than ~256 KV rows per split -- below that,
    // per-split fixed overhead (kernel launch's share, the combine
    // kernel's per-split read) starts to dominate the real compute.
    int max_useful_splits = std::max(1, N / 256);
    return std::min(target_splits, max_useful_splits);
}

// num_kv_splits explicit rather than auto-chosen — the bench uses this to
// compare splits=1 (equivalent to no split-K at all: one block per
// (batch,head), serially looping the whole KV sequence, the decode-time
// degenerate case of flash_attn_fwd's grid) against the auto-chosen split
// count, isolating exactly this technique's effect.
template <int D, int kThreadsPerBlock = 128>
inline void launch_flash_decoding_explicit_splits(
    const __half* Q, const __half* K, const __half* V, __half* O,
    float* workspace_partial_O, float* workspace_partial_m, float* workspace_partial_l,
    int B, int H, int N, int num_kv_splits, cudaStream_t stream = 0)
{
    const int BH = B * H;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    dim3 grid1(BH, num_kv_splits), block1(kThreadsPerBlock);
    flash_decoding_partial<D, kThreadsPerBlock><<<grid1, block1, 0, stream>>>(
        Q, K, V, workspace_partial_O, workspace_partial_m, workspace_partial_l, N, num_kv_splits, scale);

    dim3 grid2(BH), block2(1);
    flash_decoding_combine<D><<<grid2, block2, 0, stream>>>(
        workspace_partial_O, workspace_partial_m, workspace_partial_l, O, num_kv_splits);
}

template <int D, int kThreadsPerBlock = 128>
inline void launch_flash_decoding(
    const __half* Q, const __half* K, const __half* V, __half* O,
    float* workspace_partial_O, float* workspace_partial_m, float* workspace_partial_l,
    int B, int H, int N, cudaStream_t stream = 0)
{
    const int BH = B * H;
    const int num_kv_splits = choose_num_kv_splits(B, H, N);
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    dim3 grid1(BH, num_kv_splits), block1(kThreadsPerBlock);
    flash_decoding_partial<D, kThreadsPerBlock><<<grid1, block1, 0, stream>>>(
        Q, K, V, workspace_partial_O, workspace_partial_m, workspace_partial_l, N, num_kv_splits, scale);

    dim3 grid2(BH), block2(1);
    flash_decoding_combine<D><<<grid2, block2, 0, stream>>>(
        workspace_partial_O, workspace_partial_m, workspace_partial_l, O, num_kv_splits);
}
