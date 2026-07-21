// data_parallel_test.cpp — validates PLAN.md step 3's requirement directly:
// "manual gradient all-reduce after backward pass, validate loss curve
// matches single-GPU baseline." Single-process full-batch GD is the
// "single-GPU baseline"; 4 simulated ranks (real TCP loopback threads,
// networking/ring_allreduce for the all-reduce) split the same dataset and
// must produce the same loss curve.
#include "linreg.h"

#include "ring_allreduce.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <vector>

using namespace distributed_training;

int main() {
  constexpr int kNumFeatures = 8;
  constexpr int kNumSamples = 400;
  constexpr int kWorldSize = 4;
  constexpr int kSteps = 50;
  constexpr float kLr = 0.01f;
  constexpr uint16_t kBasePort = 35101;

  std::vector<float> w_true(kNumFeatures);
  for (int j = 0; j < kNumFeatures; ++j) w_true[static_cast<size_t>(j)] = 0.5f + 0.1f * static_cast<float>(j);

  Dataset ds = make_synthetic_regression(kNumSamples, kNumFeatures, w_true, /*noise_std=*/0.05f, /*seed=*/42);

  std::vector<float> w0(kNumFeatures, 0.0f);

  // Single-process "single-GPU" baseline: full-batch GD over the whole dataset.
  std::vector<float> baseline_w = w0;
  std::vector<float> baseline_loss(kSteps);
  for (int step = 0; step < kSteps; ++step) {
    baseline_loss[static_cast<size_t>(step)] = mse_loss(ds, baseline_w);
    auto grad_sum = mse_gradient_sum(ds, baseline_w);
    for (int j = 0; j < kNumFeatures; ++j) {
      baseline_w[static_cast<size_t>(j)] -= kLr * grad_sum[static_cast<size_t>(j)] / kNumSamples;
    }
  }

  // Data-parallel: kWorldSize ranks, each owning a contiguous shard of the
  // SAME dataset, gradients all-reduced (summed, then divided by the global
  // sample count — not the per-shard count) each step.
  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  int shard_size = kNumSamples / kWorldSize;

  std::vector<std::future<std::vector<float>>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() -> std::vector<float> {
      Dataset shard{shard_size, kNumFeatures, {}, {}};
      shard.X.assign(ds.X.begin() + static_cast<long>(r) * shard_size * kNumFeatures,
                      ds.X.begin() + static_cast<long>(r + 1) * shard_size * kNumFeatures);
      shard.y.assign(ds.y.begin() + static_cast<long>(r) * shard_size, ds.y.begin() + static_cast<long>(r + 1) * shard_size);

      std::vector<float> w = w0;
      std::vector<float> loss_curve(kSteps);
      for (int step = 0; step < kSteps; ++step) {
        loss_curve[static_cast<size_t>(step)] = mse_loss(ds, w); // full-dataset loss, for comparison to baseline
        auto grad = mse_gradient_sum(shard, w); // local partial sum
        ring_allreduce(grad.data(), grad.size(), *ch); // now the global sum, on every rank
        for (int j = 0; j < kNumFeatures; ++j) {
          w[static_cast<size_t>(j)] -= kLr * grad[static_cast<size_t>(j)] / kNumSamples;
        }
      }
      return r == 0 ? loss_curve : std::vector<float>{};
    }));
  }

  std::vector<float> rank0_loss;
  for (int r = 0; r < kWorldSize; ++r) {
    auto lc = results[static_cast<size_t>(r)].get();
    if (r == 0) rank0_loss = std::move(lc);
  }

  bool ok = true;
  std::printf("step  baseline_loss  data_parallel_loss  rel_diff\n");
  for (int step = 0; step < kSteps; step += 10) {
    float b = baseline_loss[static_cast<size_t>(step)];
    float d = rank0_loss[static_cast<size_t>(step)];
    float rel_diff = std::abs(b - d) / std::max(1e-6f, std::abs(b));
    std::printf("%4d  %13.6f  %18.6f  %8.5f\n", step, b, d, rel_diff);
    if (rel_diff > 1e-2f) ok = false;
  }
  std::printf("final loss: baseline=%.6f data_parallel=%.6f\n", baseline_loss.back(), rank0_loss.back());
  if (baseline_loss.back() > baseline_loss.front() * 0.5f) {
    std::printf("baseline did not converge meaningfully — test setup problem\n");
    ok = false;
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
