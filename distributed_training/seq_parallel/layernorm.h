#pragma once

// LayerNorm, sharded by SEQUENCE position: PLAN.md step 13. LayerNorm
// normalizes each row (token) independently across the hidden dimension,
// so sharding by sequence needs zero communication — a rank just runs
// this same forward/backward on whichever rows it owns. The interesting
// part of sequence parallelism isn't this file, it's the BOUNDARY with
// tensor parallelism (see seq_parallel_test.cpp): entering a
// tensor-parallel region needs the full sequence (an all-gather), and
// leaving one can produce a sequence SHARD directly via reduce-scatter
// instead of an all-reduce followed by a manual re-shard — same
// communication volume as an all-reduce for the sum itself, but each rank
// only receives its own 1/world_size slice instead of the full result,
// which is the actual memory point of combining the two parallelism
// strategies this way.

#include <cmath>
#include <vector>

#include "matrix.h"

namespace distributed_training {

struct LayerNormCache {
  Matrix centered; // x - mean, per row
  std::vector<float> inv_std; // 1/sqrt(var+eps), per row
  Matrix gamma;
};

inline Matrix layernorm_forward(const Matrix &x, const Matrix &gamma, const Matrix &beta, LayerNormCache &cache,
                                 float eps = 1e-5f) {
  int n = x.rows(), h = x.cols();
  Matrix out(n, h);
  Matrix centered(n, h);
  std::vector<float> inv_std(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    float mean = 0.0f;
    for (int j = 0; j < h; ++j) mean += x(i, j);
    mean /= static_cast<float>(h);
    float var = 0.0f;
    for (int j = 0; j < h; ++j) {
      centered(i, j) = x(i, j) - mean;
      var += centered(i, j) * centered(i, j);
    }
    var /= static_cast<float>(h);
    float istd = 1.0f / std::sqrt(var + eps);
    inv_std[static_cast<size_t>(i)] = istd;
    for (int j = 0; j < h; ++j) out(i, j) = centered(i, j) * istd * gamma(0, j) + beta(0, j);
  }
  cache = LayerNormCache{centered, inv_std, gamma};
  return out;
}

struct LayerNormGrads {
  Matrix dx, dgamma, dbeta;
};

inline LayerNormGrads layernorm_backward(const LayerNormCache &cache, const Matrix &dy) {
  int n = dy.rows(), h = dy.cols();
  Matrix dgamma(1, h), dbeta(1, h), dx(n, h);
  for (int i = 0; i < n; ++i) {
    float dxhat_sum = 0.0f, dxhat_xhat_sum = 0.0f;
    for (int j = 0; j < h; ++j) {
      float xhat = cache.centered(i, j) * cache.inv_std[static_cast<size_t>(i)];
      float dxhat = dy(i, j) * cache.gamma(0, j);
      dxhat_sum += dxhat;
      dxhat_xhat_sum += dxhat * xhat;
      dgamma(0, j) += dy(i, j) * xhat;
      dbeta(0, j) += dy(i, j);
    }
    float dxhat_mean = dxhat_sum / static_cast<float>(h);
    float dxhat_xhat_mean = dxhat_xhat_sum / static_cast<float>(h);
    for (int j = 0; j < h; ++j) {
      float xhat = cache.centered(i, j) * cache.inv_std[static_cast<size_t>(i)];
      float dxhat = dy(i, j) * cache.gamma(0, j);
      dx(i, j) = cache.inv_std[static_cast<size_t>(i)] * (dxhat - dxhat_mean - xhat * dxhat_xhat_mean);
    }
  }
  return LayerNormGrads{dx, dgamma, dbeta};
}

} // namespace distributed_training
