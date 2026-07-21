#pragma once

// Distributed gradient norm clipping: PLAN.md step 5. Written for the
// SHARDED-gradient case (each rank holds only part of the full gradient
// vector) rather than assuming every rank already has the complete
// gradient — pure data-parallel training doesn't strictly need this (the
// all-reduced gradient is already identical and complete on every rank),
// but ZeRO (steps 7-9) shards gradients across ranks, and clipping still
// has to be computed against the GLOBAL norm across all shards, not each
// rank's local slice. Building it the general way now means steps 7-9
// reuse it unchanged.

#include <algorithm>
#include <cmath>
#include <vector>

#include "ring_allreduce.h"

namespace distributed_training {

// Computes ||g||_2 across the full (possibly sharded) gradient: each rank
// contributes its local shard's sum of squares, ring_allreduce sums those
// scalars across ranks, every rank takes the sqrt — so every rank ends up
// with the identical global norm.
inline float global_grad_norm(const std::vector<float> &local_shard, netcommon::Channel &channel) {
  float local_sumsq = 0.0f;
  for (float g : local_shard) local_sumsq += g * g;

  ring_allreduce(&local_sumsq, 1, channel);
  return std::sqrt(local_sumsq);
}

// Scales this rank's shard in place by min(1, max_norm / (global_norm +
// eps)) — the standard "clip by global norm" rule. Every rank must be
// called with the SAME global_norm (from global_grad_norm above) so every
// shard is scaled by the same factor, equivalent to clipping the
// unsharded gradient as one vector.
inline float clip_grad_by_global_norm(std::vector<float> &local_shard, float global_norm, float max_norm) {
  constexpr float kEps = 1e-6f;
  float scale = std::min(1.0f, max_norm / (global_norm + kEps));
  if (scale < 1.0f) {
    for (float &g : local_shard) g *= scale;
  }
  return scale;
}

} // namespace distributed_training
