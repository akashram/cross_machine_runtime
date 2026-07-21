#pragma once

// ZeRO-Infinity: offload optimizer state / parameters / gradients beyond
// GPU memory to CPU RAM or NVMe, overlapping the transfer with compute so
// it is not simply serialized in front of every step.
//
// Two independent things live here:
//  1. `simulate_offload_schedule`: an ANALYTICAL cost model (same spirit
//     as compiler/cost_model/ — see that step's README for the precedent
//     of a Mac-runnable roofline-style estimate standing in for a
//     hardware benchmark) comparing naive (transfer-then-compute,
//     serialized) vs. double-buffered (prefetch next shard's transfer
//     while computing the current one) total time, given a compute time
//     and transfer time per shard. The actual GB/s numbers for CPU RAM
//     (PCIe) and NVMe are configuration inputs, not measurements — see
//     README.md's Results table for real numbers, which need real
//     hardware.
//  2. `run_double_buffered`: the actual double-buffering SCHEDULING LOGIC,
//     executed for real (calls real transfer_fn/compute_fn callbacks in
//     prefetch order) — this is the part that is fully real, tested code,
//     independent of what the callbacks actually do. On real hardware,
//     transfer_fn issues an async DMA/cudaMemcpyAsync and compute_fn
//     launches a kernel on a different stream so they genuinely overlap;
//     here the test's callbacks just record which shard ran when, to
//     verify the schedule itself visits every shard exactly once, in
//     order, with the correct prefetch-ahead relationship — a property
//     that does not depend on the target device at all.

#include <algorithm>
#include <functional>
#include <vector>

namespace distributed_training {

enum class OffloadTier { GpuOnly, CpuRam, Nvme };

struct OffloadScheduleResult {
  double naive_total_ms;
  double overlapped_total_ms;
  double speedup;
};

// naive: every shard pays compute_time_ms + transfer_time_ms, back to back.
// overlapped: one pipeline-fill transfer, then num_shards stages each
// taking max(compute, transfer) once the two are running concurrently —
// the standard software-pipelining total-time formula.
inline OffloadScheduleResult simulate_offload_schedule(int num_shards, double compute_time_ms,
                                                        double transfer_time_ms) {
  double naive = static_cast<double>(num_shards) * (compute_time_ms + transfer_time_ms);
  double overlapped = transfer_time_ms + static_cast<double>(num_shards) * std::max(compute_time_ms, transfer_time_ms);
  return OffloadScheduleResult{naive, overlapped, naive / overlapped};
}

// Runs the real double-buffered prefetch schedule: transfer shard 0 (fill),
// then for each shard i, prefetch shard i+1's transfer before computing
// shard i (on real hardware these run concurrently on separate
// engines/streams — here they are just sequential calls whose ORDER is
// what is being validated, not their timing). Returns the order compute_fn
// was invoked in.
inline std::vector<int> run_double_buffered(int num_shards, const std::function<void(int)> &transfer_fn,
                                             const std::function<void(int)> &compute_fn) {
  std::vector<int> order;
  if (num_shards == 0) return order;
  transfer_fn(0);
  for (int i = 0; i < num_shards; ++i) {
    if (i + 1 < num_shards) transfer_fn(i + 1);
    compute_fn(i);
    order.push_back(i);
  }
  return order;
}

} // namespace distributed_training
