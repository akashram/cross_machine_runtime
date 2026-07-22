#pragma once

// SyncBatchNorm: PLAN.md step 19. BatchNorm normalizes each FEATURE
// across the BATCH dimension (unlike LayerNorm, step 13, which normalizes
// each ROW across the feature dimension) — so in data-parallel training,
// each rank only sees its own local batch, and normalizing against local-
// only statistics is a different (noisier) computation than what a single
// larger global batch would produce, especially at small per-rank batch
// sizes. SyncBatchNorm all-reduces the batch statistics across every
// data-parallel rank so every rank normalizes against the SAME global
// mean/variance.
//
// The math is structurally the same shape as LayerNorm's (see
// seq_parallel/layernorm.h) — a closed-form backward through a
// normalize-then-affine op — just reducing over a different axis: here,
// the sum is over every SAMPLE across every RANK (via all-reduce) for a
// fixed feature, instead of over every FEATURE within one row.
//
// Not optimized for round-trip count: forward issues 3 separate
// all-reduces (sum, sum-of-squares, count) and backward issues 4
// (dgamma, dbeta, and the two global reduction terms S1/S2 needed for
// dx) where a real implementation would pack them into fewer wire
// round-trips (e.g. one all-reduce over a concatenated buffer). Kept
// separate here for clarity — this step validates the DISTRIBUTED
// STATISTICS math is correct, not communication-count optimization,
// which is a real but separate follow-up (see README.md).

#include <cmath>
#include <vector>

#include "matrix.h"
#include "ring_allreduce.h"

namespace distributed_training {

struct SyncBNCache {
  Matrix xhat;
  Matrix gamma;
  std::vector<float> inv_std;
  float global_count;
};

inline Matrix sync_batchnorm_forward(const Matrix &x_local, const Matrix &gamma, const Matrix &beta,
                                      SyncBNCache &cache, netcommon::Channel &channel, float eps = 1e-5f) {
  int local_n = x_local.rows();
  int features = x_local.cols();

  std::vector<float> sum(static_cast<size_t>(features), 0.0f), sumsq(static_cast<size_t>(features), 0.0f);
  for (int r = 0; r < local_n; ++r) {
    for (int f = 0; f < features; ++f) {
      sum[static_cast<size_t>(f)] += x_local(r, f);
      sumsq[static_cast<size_t>(f)] += x_local(r, f) * x_local(r, f);
    }
  }
  float count = static_cast<float>(local_n);

  ring_allreduce(sum.data(), sum.size(), channel);
  ring_allreduce(sumsq.data(), sumsq.size(), channel);
  ring_allreduce(&count, 1, channel);

  std::vector<float> mean(static_cast<size_t>(features)), inv_std(static_cast<size_t>(features));
  for (int f = 0; f < features; ++f) {
    mean[static_cast<size_t>(f)] = sum[static_cast<size_t>(f)] / count;
    float var = sumsq[static_cast<size_t>(f)] / count - mean[static_cast<size_t>(f)] * mean[static_cast<size_t>(f)];
    inv_std[static_cast<size_t>(f)] = 1.0f / std::sqrt(var + eps);
  }

  Matrix xhat(local_n, features), out(local_n, features);
  for (int r = 0; r < local_n; ++r) {
    for (int f = 0; f < features; ++f) {
      xhat(r, f) = (x_local(r, f) - mean[static_cast<size_t>(f)]) * inv_std[static_cast<size_t>(f)];
      out(r, f) = xhat(r, f) * gamma(0, f) + beta(0, f);
    }
  }

  cache = SyncBNCache{xhat, gamma, inv_std, count};
  return out;
}

struct SyncBNGrads {
  Matrix dx, dgamma, dbeta;
};

inline SyncBNGrads sync_batchnorm_backward(const SyncBNCache &cache, const Matrix &dy_local,
                                            netcommon::Channel &channel) {
  int local_n = dy_local.rows();
  int features = dy_local.cols();

  Matrix dgamma(1, features), dbeta(1, features);
  std::vector<float> s1(static_cast<size_t>(features), 0.0f), s2(static_cast<size_t>(features), 0.0f); // S1=sum(dxhat), S2=sum(dxhat*xhat)
  for (int r = 0; r < local_n; ++r) {
    for (int f = 0; f < features; ++f) {
      dgamma(0, f) += dy_local(r, f) * cache.xhat(r, f);
      dbeta(0, f) += dy_local(r, f);
      float dxhat = dy_local(r, f) * cache.gamma(0, f);
      s1[static_cast<size_t>(f)] += dxhat;
      s2[static_cast<size_t>(f)] += dxhat * cache.xhat(r, f);
    }
  }

  ring_allreduce(dgamma.data(), dgamma.size(), channel);
  ring_allreduce(dbeta.data(), dbeta.size(), channel);
  ring_allreduce(s1.data(), s1.size(), channel);
  ring_allreduce(s2.data(), s2.size(), channel);

  Matrix dx(local_n, features);
  for (int r = 0; r < local_n; ++r) {
    for (int f = 0; f < features; ++f) {
      float dxhat = dy_local(r, f) * cache.gamma(0, f);
      dx(r, f) = cache.inv_std[static_cast<size_t>(f)] *
                 (dxhat - s1[static_cast<size_t>(f)] / cache.global_count -
                  cache.xhat(r, f) * s2[static_cast<size_t>(f)] / cache.global_count);
    }
  }

  return SyncBNGrads{dx, dgamma, dbeta};
}

} // namespace distributed_training
