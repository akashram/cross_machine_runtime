#pragma once

// Block-level shared memory primitives
// =========================================================================
//
// These build on the warp-level primitives in warp_ops.h to provide
// block-level operations. Where warp_ops.h uses only shuffle instructions
// (no memory, no barriers), the block-level equivalents need shared memory
// for inter-warp communication — but only ONE __syncthreads() round-trip
// for the inter-warp step, not one per tree level.
//
//
// API contract
// ------------
// All functions take a `T* scratch` pointer to caller-allocated shared
// memory. The caller must:
//   - Allocate at least (blockDim.x / 32) elements of type T in shared mem
//   - Not use that memory for anything else during this call
//
// Why caller-allocated rather than `extern __shared__`?
// If the function declared its own `extern __shared__` it would alias with
// the caller's shared memory, producing undefined behaviour. Passing a
// pointer makes the allocation explicit and composable.
//
//
// Block-level inclusive scan — three-phase algorithm
// ---------------------------------------------------
// Phase 1 (warp scope):  warp::scan_inclusive — 5 shfl_up, no barriers
// Phase 2 (block scope): lane 31 of each warp writes its warp-total to
//                        scratch[]; one __syncthreads(); first warp scans
//                        the warp-totals using warp::scan_inclusive; writes
//                        the warp prefix sums back to scratch[]
// Phase 3 (all warps):   each thread adds scratch[warp_id - 1] to its
//                        intra-warp result — one __syncthreads()
//
// Total: 2 × __syncthreads(), 5 + 5 shfl_up instructions.
// Pure-shared-memory Hillis-Steele scan: log2(N) × __syncthreads() barriers.
// For N=256 (8 warps): 2 barriers vs 8. For N=1024: 2 vs 10.
//
//
// Bank conflicts in shared memory
// --------------------------------
// Banks are 4 bytes wide, indexed 0–31. Address A maps to bank (A/4) % 32.
// For float scratch[K]: scratch[i] maps to bank i % 32.
// With K ≤ 32 (warp count ≤ 32 → blockDim.x ≤ 1024), scratch access is:
//   - Phase 2 write: lane 0 of warp i writes scratch[i] — one thread writes,
//     no conflict.
//   - Phase 3 read: all threads in warp i read scratch[i-1] — broadcast
//     (same address, single bank) — CUDA hardware services as one transaction.
// Conclusion: the inter-warp scratch accesses are conflict-free.

#include "gpu_engine/warp_primitives/warp_ops.h"

namespace gpu_engine {
namespace block {

// -----------------------------------------------------------------------
// block_scan_inclusive
// Returns the inclusive prefix sum of `val` across all threads in the block.
// blockDim.x must be a multiple of 32.
// scratch must have at least (blockDim.x / 32) elements.
// -----------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ T block_scan_inclusive(T val, T* scratch) {
    const int lane    = warp::lane_id();
    const int warp_id = threadIdx.x / 32;
    const int n_warps = blockDim.x / 32;

    // Phase 1: intra-warp inclusive scan (no barriers)
    T intra = warp::scan_inclusive(val);

    // Phase 2: warp totals → scratch → scan the totals
    // Lane 31 of each warp holds the warp's total after Phase 1.
    if (lane == 31) scratch[warp_id] = intra;
    __syncthreads();

    if (warp_id == 0) {
        // First warp scans the n_warps warp-totals.
        // Lanes ≥ n_warps contribute 0 so the scan is over exactly n_warps values.
        T wt = (lane < n_warps) ? scratch[lane] : T(0);
        wt = warp::scan_inclusive(wt);
        if (lane < n_warps) scratch[lane] = wt;
    }
    __syncthreads();

    // Phase 3: add this warp's prefix (= inclusive sum of all previous warps)
    // scratch[warp_id] now holds the inclusive sum of warps 0..warp_id.
    // The prefix for warp_id's threads is scratch[warp_id - 1] (exclusive of this warp).
    T warp_prefix = (warp_id > 0) ? scratch[warp_id - 1] : T(0);
    return intra + warp_prefix;
}

// -----------------------------------------------------------------------
// block_scan_exclusive
// Returns the exclusive prefix sum: thread i gets sum(val[0]..val[i-1]).
// Thread 0 returns 0 (identity).
// -----------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ T block_scan_exclusive(T val, T* scratch) {
    // Inclusive scan then subtract original value (correct for addition).
    return block_scan_inclusive(val, scratch) - val;
}

// -----------------------------------------------------------------------
// block_reduce_sum
// All threads return the block-wide sum via scratch[0].
// (All threads must call this — it calls __syncthreads internally.)
// scratch must have at least (blockDim.x / 32) elements.
// -----------------------------------------------------------------------
template<typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* scratch) {
    const int lane    = warp::lane_id();
    const int warp_id = threadIdx.x / 32;
    const int n_warps = blockDim.x / 32;

    // Intra-warp reduce — only lane 0 gets result
    val = warp::reduce_sum_lane0(val);

    if (lane == 0) scratch[warp_id] = val;
    __syncthreads();

    // First warp reduces the n_warps partial sums
    if (warp_id == 0) {
        val = (lane < n_warps) ? scratch[lane] : T(0);
        val = warp::reduce_sum_lane0(val);
        if (lane == 0) scratch[0] = val;
    }
    __syncthreads();

    return scratch[0];  // all threads return the same value
}

} // namespace block
} // namespace gpu_engine
