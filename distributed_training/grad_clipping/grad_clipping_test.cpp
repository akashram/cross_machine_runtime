// grad_clipping_test.cpp — a full gradient vector, sharded across 4
// simulated ranks (mimicking ZeRO-style gradient sharding), must clip to
// the exact same result as clipping the unsharded vector in one process:
// (1) an above-threshold case that actually clips, (2) a below-threshold
// case that's a no-op.
#include "grad_clipping.h"

#include <cmath>
#include <cstdio>
#include <future>
#include <random>
#include <vector>

using namespace distributed_training;

namespace {

bool run_case(const char *name, float max_norm, uint32_t seed, float scale_hint) {
  constexpr int kWorldSize = 4;
  constexpr int kTotalParams = 4000; // 1000/rank
  constexpr uint16_t kBasePort = 35201;
  const int shard_size = kTotalParams / kWorldSize;

  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> full_grad(kTotalParams);
  for (float &g : full_grad) g = dist(rng) * scale_hint;

  // Single-process reference.
  double sumsq = 0.0;
  for (float g : full_grad) sumsq += static_cast<double>(g) * g;
  float true_norm = static_cast<float>(std::sqrt(sumsq));
  float true_scale = std::min(1.0f, max_norm / (true_norm + 1e-6f));
  std::vector<float> expected(kTotalParams);
  for (int i = 0; i < kTotalParams; ++i) expected[static_cast<size_t>(i)] = full_grad[static_cast<size_t>(i)] * true_scale;

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::vector<std::future<std::vector<float>>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() -> std::vector<float> {
      std::vector<float> shard(full_grad.begin() + static_cast<long>(r) * shard_size,
                                full_grad.begin() + static_cast<long>(r + 1) * shard_size);
      float gnorm = global_grad_norm(shard, *ch);
      clip_grad_by_global_norm(shard, gnorm, max_norm);
      return shard;
    }));
  }

  std::vector<float> distributed_result;
  for (auto &f : results) {
    auto shard = f.get();
    distributed_result.insert(distributed_result.end(), shard.begin(), shard.end());
  }

  bool ok = true;
  float max_abs_diff = 0.0f;
  for (int i = 0; i < kTotalParams; ++i) {
    float diff = std::abs(expected[static_cast<size_t>(i)] - distributed_result[static_cast<size_t>(i)]);
    max_abs_diff = std::max(max_abs_diff, diff);
    if (diff > 1e-3f) ok = false;
  }
  std::printf("%s: true_norm=%.4f max_norm=%.4f scale=%.4f max_abs_diff=%.6f: %s\n", name, true_norm, max_norm,
              true_scale, max_abs_diff, ok ? "PASS" : "FAIL");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  // Gradient scaled up (scale_hint=5) so its norm comfortably exceeds max_norm=10 -> clips.
  ok = run_case("above-threshold (clips)", /*max_norm=*/10.0f, /*seed=*/1, /*scale_hint=*/5.0f) && ok;
  // Gradient scaled down (scale_hint=0.01) so its norm is comfortably under max_norm=10 -> no-op.
  ok = run_case("below-threshold (no-op)", /*max_norm=*/10.0f, /*seed=*/2, /*scale_hint=*/0.01f) && ok;

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
