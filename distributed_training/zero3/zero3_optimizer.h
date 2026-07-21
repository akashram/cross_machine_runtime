#pragma once

// ZeRO Stage 3: shard PARAMETERS too, on top of step 8's gradient and
// optimizer-state sharding. The defining property, and the one this file
// actually validates: a rank's PERSISTENT (between-step) memory holds only
// its 1/world_size shard of parameters, gradients, and optimizer state —
// never the full model. The full parameter vector exists only
// transiently, gathered fresh at the start of each step and discarded once
// that step's forward/backward/update is done.
//
// What this simplifies vs. a real ZeRO-3 implementation: real ZeRO-3
// gathers and releases parameters PER LAYER, just before that layer's
// forward (and again just before its backward), so PEAK memory during a
// single step stays near 1/world_size too, not just the between-step
// floor. This class gathers the WHOLE parameter vector once per step
// instead (simpler, and correctness-equivalent — the training math doesn't
// care when within a step the gather happens) — it validates ZeRO-3's
// correctness contract and its between-step memory floor, not its
// per-layer peak-memory behavior, which only matters at real model scale
// with real memory pressure. See README.md's Results table.

#include <cstddef>
#include <vector>

#include "../zero1/adam.h"
#include "collectives.h"

namespace distributed_training {

class ZeroStage3Optimizer {
public:
  ZeroStage3Optimizer(size_t total_params, int rank, int world_size, float lr = 1e-3f, float beta1 = 0.9f,
                       float beta2 = 0.999f)
      : world_size_(world_size), lr_(lr), beta1_(beta1), beta2_(beta2),
        shard_size_((total_params + static_cast<size_t>(world_size) - 1) / static_cast<size_t>(world_size)),
        owned_shard_(static_cast<size_t>((rank + 1) % world_size)), params_shard_(shard_size_), state_(shard_size_) {
  }

  size_t padded_total_params() const { return shard_size_ * static_cast<size_t>(world_size_); }
  size_t shard_size() const { return shard_size_; }
  size_t owned_shard_index() const { return owned_shard_; }

  // Seeds this rank's persistent shard from a full initial parameter
  // vector (called once, at model init — every rank passes the SAME
  // full_params, e.g. broadcast from rank 0 in a real system).
  void init_from_full(const std::vector<float> &full_params) {
    size_t start = owned_shard_ * shard_size_;
    params_shard_.assign(full_params.begin() + static_cast<long>(start),
                          full_params.begin() + static_cast<long>(start + shard_size_));
  }

  // Transiently materializes the full parameter vector for this step's
  // forward/backward. Caller must not assume this stays valid or correct
  // past the current step (it is not stored — it is rebuilt from
  // params_shard_ across all ranks every time this is called).
  std::vector<float> gather_full_params(netcommon::Channel &channel) const {
    std::vector<float> full(padded_total_params());
    // Need a non-const copy of params_shard_ for AllGather's send_buf even
    // though it only reads it — kept explicit rather than const_cast.
    std::vector<float> send = params_shard_;
    collectives::AllGather(send.data(), shard_size_, full.data(), channel);
    return full;
  }

  // local_grad must be this rank's own LOCAL (un-reduced) gradient, full
  // length (padded_total_params()) — reduce-scattered in place, same
  // contract as ZeroStage2Optimizer::step. Updates params_shard_ in place;
  // does NOT reassemble a full parameter vector (that is
  // gather_full_params's job, called fresh next step).
  void step(std::vector<float> &local_grad, netcommon::Channel &channel) {
    collectives::ReduceScatter(local_grad.data(), local_grad.size(), channel);
    size_t start = owned_shard_ * shard_size_;
    std::vector<float> owned_grad(local_grad.begin() + static_cast<long>(start),
                                   local_grad.begin() + static_cast<long>(start + shard_size_));
    adam_step(params_shard_, owned_grad, state_, lr_, beta1_, beta2_);
  }

private:
  int world_size_;
  float lr_;
  float beta1_;
  float beta2_;
  size_t shard_size_;
  size_t owned_shard_;
  std::vector<float> params_shard_;
  AdamState state_;
};

} // namespace distributed_training
