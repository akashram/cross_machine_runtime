#pragma once

// ZeRO Stage 1: shard OPTIMIZER STATE (Adam's m/v, 2x parameter memory)
// across data-parallel ranks, while parameters and gradients stay fully
// replicated — the cheapest ZeRO stage to add on top of plain data
// parallelism (steps 3-5), and the one step 8/9 (ZeRO-2/3) shard
// progressively more onto.
//
// Contract each rank must uphold, every step, in order:
//   1. Compute a LOCAL gradient (e.g. from its own data shard, as in
//      step 3) and all-reduce it so `full_grad` is the complete, correctly-
//      scaled, identical-on-every-rank gradient (unchanged from plain data
//      parallel — see data_parallel/README.md).
//   2. Call step(full_params, full_grad, channel): this rank updates ONLY
//      its own shard of full_params in place, using ONLY its own Adam
//      state (never materializing another rank's m/v — that memory
//      saving is the entire point) — then all-gathers so every rank ends
//      the call with the SAME, fully updated full_params, ready for next
//      step's forward pass. Every rank must call step() with the same
//      full_grad (already all-reduced) or the shards diverge.
//
// Padding: collectives::AllGather requires a uniform send_count across
// ranks, so total_params is padded up to a multiple of world_size; the
// padding tail is never read by anything that reconstructs the real
// parameters (see mlp.h's unflatten_params).

#include <algorithm>
#include <cstddef>
#include <vector>

#include "adam.h"
#include "collectives.h"

namespace distributed_training {

class ZeroStage1Optimizer {
public:
  ZeroStage1Optimizer(size_t total_params, int rank, int world_size, float lr = 1e-3f, float beta1 = 0.9f,
                       float beta2 = 0.999f)
      : rank_(rank), world_size_(world_size), lr_(lr), beta1_(beta1), beta2_(beta2),
        shard_size_((total_params + static_cast<size_t>(world_size) - 1) / static_cast<size_t>(world_size)),
        state_(shard_size_) {}

  size_t padded_total_params() const { return shard_size_ * static_cast<size_t>(world_size_); }
  size_t shard_size() const { return shard_size_; }

  // full_params/full_grad must be padded_total_params() long on every rank
  // (see mlp.h's flatten_params — callers pad by resizing after flattening).
  void step(std::vector<float> &full_params, const std::vector<float> &full_grad, netcommon::Channel &channel) {
    size_t shard_start = static_cast<size_t>(rank_) * shard_size_;

    std::vector<float> local_param_shard(full_params.begin() + static_cast<long>(shard_start),
                                          full_params.begin() + static_cast<long>(shard_start + shard_size_));
    std::vector<float> local_grad_shard(full_grad.begin() + static_cast<long>(shard_start),
                                         full_grad.begin() + static_cast<long>(shard_start + shard_size_));

    adam_step(local_param_shard, local_grad_shard, state_, lr_, beta1_, beta2_);

    // AllGather's contract: rank r's contribution lands at slot (r+1)%N in
    // recv_buf (see collectives.h), a rotation inherited from the ring
    // all-gather it is built on. Un-rotate when writing back into
    // full_params so shard i still corresponds to params
    // [i*shard_size, (i+1)*shard_size) for every rank, matching how the
    // shard was read out above.
    std::vector<float> gathered(shard_size_ * static_cast<size_t>(world_size_));
    collectives::AllGather(local_param_shard.data(), shard_size_, gathered.data(), channel);

    for (int owner = 0; owner < world_size_; ++owner) {
      size_t slot = static_cast<size_t>((owner + 1) % world_size_);
      std::copy(gathered.begin() + static_cast<long>(slot * shard_size_),
                gathered.begin() + static_cast<long>((slot + 1) * shard_size_),
                full_params.begin() + static_cast<long>(static_cast<size_t>(owner) * shard_size_));
    }
  }

private:
  int rank_;
  int world_size_;
  float lr_;
  float beta1_;
  float beta2_;
  size_t shard_size_;
  AdamState state_;
};

} // namespace distributed_training
