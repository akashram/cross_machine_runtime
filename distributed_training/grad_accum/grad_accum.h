#pragma once

// Gradient accumulation: PLAN.md step 4. Splits a logical batch into
// micro-batches (fits in memory/compute budget one at a time), sums their
// gradients, and applies one optimizer step scaled by the TOTAL sample
// count across all micro-batches — not by the micro-batch count and not by
// a single micro-batch's size.
//
// The bug this exists to prevent: accumulating micro-batch MEANS (dividing
// by each micro-batch's own size before summing) instead of micro-batch
// SUMS silently gives the wrong answer whenever micro-batches are uneven
// sized (the last one in a data loader is almost always smaller), and even
// for even-sized micro-batches, forgetting to divide by
// (num_microbatches * microbatch_size) at all — instead by just
// microbatch_size — inflates the effective learning rate by
// num_microbatches. Both are real, easy mistakes; `GradientAccumulator`
// makes the correct scaling structural rather than a comment to remember.

#include <cstddef>
#include <vector>

namespace distributed_training {

class GradientAccumulator {
public:
  explicit GradientAccumulator(size_t num_params) : sum_(num_params, 0.0f) {}

  // Adds one micro-batch's gradient SUM (not mean — see linreg.h's
  // mse_gradient_sum) and its sample count.
  void accumulate(const std::vector<float> &grad_sum, int num_samples) {
    for (size_t i = 0; i < sum_.size(); ++i) sum_[i] += grad_sum[i];
    total_samples_ += num_samples;
  }

  // Returns the correctly-scaled gradient (sum / total samples across every
  // micro-batch accumulated since the last reset) and resets for the next
  // accumulation window.
  std::vector<float> finalize_and_reset() {
    std::vector<float> result(sum_.size());
    for (size_t i = 0; i < sum_.size(); ++i) result[i] = sum_[i] / static_cast<float>(total_samples_);
    std::fill(sum_.begin(), sum_.end(), 0.0f);
    total_samples_ = 0;
    return result;
  }

  int total_samples() const { return total_samples_; }

private:
  std::vector<float> sum_;
  int total_samples_ = 0;
};

} // namespace distributed_training
