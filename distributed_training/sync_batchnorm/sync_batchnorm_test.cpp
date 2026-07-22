// sync_batchnorm_test.cpp — the reference here is sync_batchnorm_forward/
// backward itself, called with world_size=1 over the full concatenated
// batch (a 1-rank "mesh" makes ring_allreduce a no-op, so this reduces to
// plain BatchNorm over the global batch with zero risk of a second,
// independently-derived reference formula drifting from the real one —
// same function, different rank count). 4 simulated ranks, each with a
// LOCAL shard of that same batch, must reproduce it exactly.
#include "sync_batchnorm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <random>
#include <vector>

using namespace distributed_training;

namespace {

Matrix row_slice(const Matrix &m, int start, int count) {
  Matrix out(count, m.cols());
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < m.cols(); ++j) out(i, j) = m(start + i, j);
  return out;
}

float max_abs_diff(const Matrix &a, const Matrix &b) {
  float d = 0.0f;
  for (int i = 0; i < a.rows(); ++i)
    for (int j = 0; j < a.cols(); ++j) d = std::max(d, std::abs(a(i, j) - b(i, j)));
  return d;
}

} // namespace

int main() {
  constexpr int kWorldSize = 4;
  constexpr int kLocalBatch = 5;
  constexpr int kTotalBatch = kWorldSize * kLocalBatch;
  constexpr int kFeatures = 6;
  constexpr uint16_t kBasePort = 36101;
  constexpr uint16_t kRefPort = 36151;

  std::mt19937 rng(44);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  auto random_matrix = [&](int r, int c) {
    Matrix m(r, c);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) m(i, j) = dist(rng);
    return m;
  };

  Matrix x_global = random_matrix(kTotalBatch, kFeatures);
  Matrix gamma(1, kFeatures), beta(1, kFeatures);
  for (int j = 0; j < kFeatures; ++j) { gamma(0, j) = 1.0f + 0.1f * dist(rng); beta(0, j) = 0.1f * dist(rng); }

  // --- Reference: world_size=1 "distributed" run over the full batch.
  auto ref_channel = netcommon::make_tcp_loopback_mesh(1, kRefPort);
  SyncBNCache ref_cache;
  Matrix ref_out = sync_batchnorm_forward(x_global, gamma, beta, ref_cache, *ref_channel[0]);
  Matrix ref_dy = ref_out; // loss = 0.5*sum(out^2)
  auto ref_grads = sync_batchnorm_backward(ref_cache, ref_dy, *ref_channel[0]);

  // --- Distributed: 4 ranks, each with a local shard of x_global.
  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  struct RankResult {
    Matrix out, dx, dgamma, dbeta;
  };
  std::vector<std::future<RankResult>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() -> RankResult {
      Matrix x_local = row_slice(x_global, r * kLocalBatch, kLocalBatch);
      SyncBNCache cache;
      Matrix out = sync_batchnorm_forward(x_local, gamma, beta, cache, *ch);
      Matrix dy = out; // same loss as reference
      auto grads = sync_batchnorm_backward(cache, dy, *ch);
      return RankResult{out, grads.dx, grads.dgamma, grads.dbeta};
    }));
  }

  Matrix out_reassembled(kTotalBatch, kFeatures), dx_reassembled(kTotalBatch, kFeatures);
  Matrix dgamma_rank0, dbeta_rank0;
  for (int r = 0; r < kWorldSize; ++r) {
    RankResult res = results[static_cast<size_t>(r)].get();
    for (int i = 0; i < kLocalBatch; ++i)
      for (int j = 0; j < kFeatures; ++j) {
        out_reassembled(r * kLocalBatch + i, j) = res.out(i, j);
        dx_reassembled(r * kLocalBatch + i, j) = res.dx(i, j);
      }
    if (r == 0) { dgamma_rank0 = res.dgamma; dbeta_rank0 = res.dbeta; }
  }

  bool ok = true;
  auto check = [&](const char *name, float diff, float tol = 1e-3f) {
    std::printf("  %-10s max abs diff = %.6f: %s\n", name, diff, diff <= tol ? "PASS" : "FAIL");
    if (diff > tol) ok = false;
  };
  check("forward", max_abs_diff(ref_out, out_reassembled));
  check("dx", max_abs_diff(ref_grads.dx, dx_reassembled));
  check("dgamma", max_abs_diff(ref_grads.dgamma, dgamma_rank0));
  check("dbeta", max_abs_diff(ref_grads.dbeta, dbeta_rank0));

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
