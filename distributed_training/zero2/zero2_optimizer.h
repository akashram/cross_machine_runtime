#pragma once

// ZeRO Stage 2: shard gradients in addition to optimizer state (step 7's
// ZeRO-1 sharded only optimizer state, keeping gradients fully replicated
// via all-reduce). A rank never needs to materialize the full reduced
// gradient — only its own shard — which is the whole memory-saving point.
//
// Contract, every step:
//   1. Each rank computes its OWN LOCAL (un-reduced) gradient, full length
//      (padded_total_params()), from its own data shard.
//   2. step(full_params, local_grad, channel): reduce-scatters local_grad
//      in place (collectives::ReduceScatter — every rank's local
//      contribution combines chunk-wise; each rank ends up owning exactly
//      one chunk's true sum, the rest of the buffer left as garbage —
//      hence local_grad is taken by non-const reference and destroyed).
//      This rank's owned shard is Adam-updated with its own optimizer
//      state, then all-gathered back to a full, identical parameter vector
//      on every rank.
//
// Shard-ownership convention differs from ZeRO-1 on purpose: ZeRO-1
// artificially assigns "rank r owns shard r" (arbitrary, since it reads
// from an already-fully-reduced buffer where any assignment works) and
// must UN-ROTATE collectives::AllGather's native "rank r's contribution
// lands at slot (r+1)%world_size" placement to keep that assignment.
// Here, "rank r owns shard (r+1)%world_size" is used instead — this is
// exactly the chunk collectives::ReduceScatter already gives rank r
// ownership of, AND exactly the slot collectives::AllGather places rank
// r's contribution at, so no remapping is needed in either direction (the
// mapping r -> (r+1)%world_size is a bijection over shards either way,
// just a different, but equally valid, choice of which rank owns which
// shard).

#include <cstddef>
#include <vector>

#include "../zero1/adam.h"
#include "collectives.h"

namespace distributed_training {

class ZeroStage2Optimizer {
public:
  ZeroStage2Optimizer(size_t total_params, int rank, int world_size, float lr = 1e-3f, float beta1 = 0.9f,
                       float beta2 = 0.999f)
      : world_size_(world_size), lr_(lr), beta1_(beta1), beta2_(beta2),
        shard_size_((total_params + static_cast<size_t>(world_size) - 1) / static_cast<size_t>(world_size)),
        owned_shard_(static_cast<size_t>((rank + 1) % world_size)), state_(shard_size_) {}

  size_t padded_total_params() const { return shard_size_ * static_cast<size_t>(world_size_); }
  size_t shard_size() const { return shard_size_; }

  void step(std::vector<float> &full_params, std::vector<float> &local_grad, netcommon::Channel &channel) {
    collectives::ReduceScatter(local_grad.data(), local_grad.size(), channel);

    size_t start = owned_shard_ * shard_size_;
    std::vector<float> owned_grad(local_grad.begin() + static_cast<long>(start),
                                   local_grad.begin() + static_cast<long>(start + shard_size_));
    std::vector<float> local_param_shard(full_params.begin() + static_cast<long>(start),
                                          full_params.begin() + static_cast<long>(start + shard_size_));

    adam_step(local_param_shard, owned_grad, state_, lr_, beta1_, beta2_);

    // Every rank's contribution lands at slot (r+1)%world_size == the
    // shard it corresponds to, for every rank — the mapping is a
    // bijection over [0, world_size), so `gathered` is already the
    // correctly-ordered full parameter vector with no remap needed
    // (contrast ZeroStage1Optimizer::step).
    full_params.assign(padded_total_params(), 0.0f);
    collectives::AllGather(local_param_shard.data(), shard_size_, full_params.data(), channel);
  }

private:
  int world_size_;
  float lr_;
  float beta1_;
  float beta2_;
  size_t shard_size_;
  size_t owned_shard_;
  AdamState state_;
};

} // namespace distributed_training
