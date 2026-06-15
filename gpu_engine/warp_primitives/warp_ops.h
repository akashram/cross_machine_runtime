#pragma once

// Warp-level primitives
// =========================================================================
//
// These are device-only functions that operate across a warp (32 threads)
// using hardware shuffle/ballot instructions rather than shared memory.
// They are cheaper than shared-memory equivalents because:
//
//   - No shared memory allocation or bank-conflict risk
//   - No __syncthreads() barrier (warp primitives have implicit warp-scope sync)
//   - Single-instruction latency (4–6 cycles on Volta+)
//
// All functions take a `mask` (default 0xffffffff = all 32 lanes active).
// Pass a narrower mask for sub-warp operations or when some lanes have exited.
//
//
// Reduce patterns
// ---------------
// reduce_sum / reduce_max / reduce_min use the butterfly (XOR-shift) pattern:
// all 32 lanes receive the final result. Cost: 5 shuffle instructions.
//
// reduce_sum_lane0 uses the down-shift tree: only lane 0 receives the result.
// Slightly cheaper when you only need one lane to write to global memory.
//
//
// Scan pattern
// ------------
// scan_inclusive uses Kogge-Stone: O(log2 N) steps, all lanes receive their
// prefix sum simultaneously. Cost: 5 shuffle instructions.
// scan_exclusive shifts the inclusive result right by one lane.
//
//
// Ballot
// ------
// ballot(pred) returns a 32-bit mask where bit i is set iff lane i's pred is
// true. ballot_count(pred) returns __popc(ballot(pred)).
//
// These are useful for:
//   - Compaction (stream compaction without atomic contention)
//   - Predicated operations (apply work only where pred is true)
//   - Divergence analysis (__ballot_sync reveals which lanes are active)
//
//
// Type support
// ------------
// All shuffle functions natively support: int, unsigned int, long long,
// unsigned long long, float, double. For other types, decompose into
// 32-bit or 64-bit chunks and shuffle the pieces.

#include <cuda_runtime.h>

namespace gpu_engine {
namespace warp {

static constexpr unsigned FULL_MASK = 0xffffffff;

__device__ __forceinline__ int lane_id() {
    // %laneid is a special register — cheaper than threadIdx.x & 31
    // because it avoids a load from the threadIdx register file.
    unsigned id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
    return static_cast<int>(id);
}

// -----------------------------------------------------------------------
// Reduce — all lanes receive the result (butterfly / XOR-shift pattern)
// -----------------------------------------------------------------------

template<typename T>
__device__ __forceinline__ T reduce_sum(T val, unsigned mask = FULL_MASK) {
    val += __shfl_xor_sync(mask, val, 16);
    val += __shfl_xor_sync(mask, val,  8);
    val += __shfl_xor_sync(mask, val,  4);
    val += __shfl_xor_sync(mask, val,  2);
    val += __shfl_xor_sync(mask, val,  1);
    return val;
}

template<typename T>
__device__ __forceinline__ T reduce_max(T val, unsigned mask = FULL_MASK) {
    val = max(val, __shfl_xor_sync(mask, val, 16));
    val = max(val, __shfl_xor_sync(mask, val,  8));
    val = max(val, __shfl_xor_sync(mask, val,  4));
    val = max(val, __shfl_xor_sync(mask, val,  2));
    val = max(val, __shfl_xor_sync(mask, val,  1));
    return val;
}

template<typename T>
__device__ __forceinline__ T reduce_min(T val, unsigned mask = FULL_MASK) {
    val = min(val, __shfl_xor_sync(mask, val, 16));
    val = min(val, __shfl_xor_sync(mask, val,  8));
    val = min(val, __shfl_xor_sync(mask, val,  4));
    val = min(val, __shfl_xor_sync(mask, val,  2));
    val = min(val, __shfl_xor_sync(mask, val,  1));
    return val;
}

// -----------------------------------------------------------------------
// Reduce — only lane 0 receives the result (down-shift tree)
// Use when only thread 0 needs to write the result to global memory.
// Saves the __shfl_sync(mask, val, 0) broadcast compared to butterfly.
// -----------------------------------------------------------------------

template<typename T>
__device__ __forceinline__ T reduce_sum_lane0(T val, unsigned mask = FULL_MASK) {
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val,  8);
    val += __shfl_down_sync(mask, val,  4);
    val += __shfl_down_sync(mask, val,  2);
    val += __shfl_down_sync(mask, val,  1);
    return val;
}

// -----------------------------------------------------------------------
// Broadcast — lane src's value goes to all lanes
// -----------------------------------------------------------------------

template<typename T>
__device__ __forceinline__ T broadcast(T val, int src,
                                        unsigned mask = FULL_MASK) {
    return __shfl_sync(mask, val, src);
}

// -----------------------------------------------------------------------
// Scan — Kogge-Stone inclusive prefix sum
// After this call, lane i holds sum(lane[0] ... lane[i]).
// Cost: 5 shfl_up + 5 conditional adds (no barriers, no shared memory).
// -----------------------------------------------------------------------

template<typename T>
__device__ __forceinline__ T scan_inclusive(T val,
                                             unsigned mask = FULL_MASK) {
    const int lane = lane_id();
    T tmp;
    tmp = __shfl_up_sync(mask, val,  1); if (lane >=  1) val += tmp;
    tmp = __shfl_up_sync(mask, val,  2); if (lane >=  2) val += tmp;
    tmp = __shfl_up_sync(mask, val,  4); if (lane >=  4) val += tmp;
    tmp = __shfl_up_sync(mask, val,  8); if (lane >=  8) val += tmp;
    tmp = __shfl_up_sync(mask, val, 16); if (lane >= 16) val += tmp;
    return val;
}

// Exclusive prefix sum: lane i holds sum(lane[0] ... lane[i-1]).
// Lane 0 holds 0 (identity for addition).
template<typename T>
__device__ __forceinline__ T scan_exclusive(T val,
                                             unsigned mask = FULL_MASK) {
    // Compute inclusive scan first, then shift right by one lane.
    T incl = scan_inclusive(val, mask);
    T excl = __shfl_up_sync(mask, incl, 1);
    return lane_id() == 0 ? T(0) : excl;
}

// -----------------------------------------------------------------------
// Ballot / predicate operations
// -----------------------------------------------------------------------

// Returns a 32-bit bitmask: bit i set iff lane i's pred is nonzero.
__device__ __forceinline__ unsigned ballot(bool pred,
                                            unsigned mask = FULL_MASK) {
    return __ballot_sync(mask, pred);
}

// Count lanes where pred is true.
__device__ __forceinline__ int ballot_count(bool pred,
                                             unsigned mask = FULL_MASK) {
    return __popc(__ballot_sync(mask, pred));
}

// True iff any active lane has pred true.
__device__ __forceinline__ bool any(bool pred, unsigned mask = FULL_MASK) {
    return __any_sync(mask, pred);
}

// True iff all active lanes have pred true.
__device__ __forceinline__ bool all(bool pred, unsigned mask = FULL_MASK) {
    return __all_sync(mask, pred);
}

} // namespace warp
} // namespace gpu_engine
