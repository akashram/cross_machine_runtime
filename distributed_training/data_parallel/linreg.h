#pragma once

// Minimal linear-regression model shared by steps 3-5 (data parallel
// baseline, grad accum, grad clipping). PLAN.md explicitly calls step 3 a
// "baseline" with "manual gradient" — no autograd engine yet (that's step
// 6) — so this is closed-form: y = X.w, MSE loss, gradient computed by
// hand rather than through a tape.
//
// Deliberately global-batch (full-batch gradient descent, not mini-batch
// SGD): the point of these three steps is validating the *distributed
// mechanics* (all-reduce correctness, accumulation correctness, clipping
// correctness) against a known-correct single-process reference, and
// full-batch keeps that reference bit-comparable — the data-parallel
// gradient after all-reduce is mathematically the average of per-shard
// gradients, which for a full-batch loss is exactly the single-process
// full-batch gradient, no stochasticity to launder the comparison.

#include <cstddef>
#include <random>
#include <vector>

namespace distributed_training {

struct Dataset {
  int num_samples;
  int num_features;
  std::vector<float> X; // row-major [num_samples x num_features]
  std::vector<float> y; // [num_samples]
};

// Generates y = X.w_true + noise for a known w_true, so the optimum is
// known and convergence can be checked, not just "loss goes down."
inline Dataset make_synthetic_regression(int num_samples, int num_features,
                                          const std::vector<float> &w_true, float noise_std,
                                          uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> feat_dist(0.0f, 1.0f);
  std::normal_distribution<float> noise_dist(0.0f, noise_std);

  Dataset ds{num_samples, num_features, {}, {}};
  ds.X.resize(static_cast<size_t>(num_samples) * num_features);
  ds.y.resize(static_cast<size_t>(num_samples));

  for (int i = 0; i < num_samples; ++i) {
    float pred = 0.0f;
    for (int j = 0; j < num_features; ++j) {
      float x = feat_dist(rng);
      ds.X[static_cast<size_t>(i) * num_features + j] = x;
      pred += x * w_true[static_cast<size_t>(j)];
    }
    ds.y[static_cast<size_t>(i)] = pred + noise_dist(rng);
  }
  return ds;
}

inline float mse_loss(const Dataset &ds, const std::vector<float> &w) {
  double total = 0.0;
  for (int i = 0; i < ds.num_samples; ++i) {
    double pred = 0.0;
    for (int j = 0; j < ds.num_features; ++j) {
      pred += static_cast<double>(ds.X[static_cast<size_t>(i) * ds.num_features + j]) * w[static_cast<size_t>(j)];
    }
    double diff = pred - ds.y[static_cast<size_t>(i)];
    total += diff * diff;
  }
  return static_cast<float>(total / ds.num_samples);
}

// Gradient of mean-squared-error w.r.t. w, over `ds` (may be a shard —
// callers doing distributed training divide by the GLOBAL sample count
// after all-reducing the SUM of per-shard sums, not the per-shard mean,
// so shards of unequal size still combine correctly).
inline std::vector<float> mse_gradient_sum(const Dataset &ds, const std::vector<float> &w) {
  std::vector<float> grad(static_cast<size_t>(ds.num_features), 0.0f);
  for (int i = 0; i < ds.num_samples; ++i) {
    double pred = 0.0;
    const float *row = &ds.X[static_cast<size_t>(i) * ds.num_features];
    for (int j = 0; j < ds.num_features; ++j) pred += static_cast<double>(row[j]) * w[static_cast<size_t>(j)];
    double err = pred - ds.y[static_cast<size_t>(i)];
    for (int j = 0; j < ds.num_features; ++j) {
      grad[static_cast<size_t>(j)] += static_cast<float>(2.0 * err * row[j]);
    }
  }
  return grad; // NOT yet divided by sample count — see divide-after-allreduce note above
}

} // namespace distributed_training
