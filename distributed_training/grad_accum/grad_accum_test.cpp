// grad_accum_test.cpp — three checks:
//  1. accumulating uneven micro-batches matches one full-batch gradient
//     exactly (correct scaling: divide by total samples across the window).
//  2. the naive-wrong scaling (divide by micro-batch COUNT instead of total
//     SAMPLES) is demonstrably a different, wrong number — the bug this
//     component exists to prevent, made concrete.
//  3. accumulation composes correctly with data parallelism: 4 ranks, each
//     accumulating 2 uneven micro-batches before one ring_allreduce, must
//     still match the single-process baseline loss curve (same property
//     step 3 validated, now with accumulation layered in).
#include "grad_accum.h"
#include "../data_parallel/linreg.h"

#include "ring_allreduce.h"

#include <cmath>
#include <cstdio>
#include <future>
#include <vector>

using namespace distributed_training;

namespace {

Dataset slice(const Dataset &ds, int start, int count) {
  Dataset out{count, ds.num_features, {}, {}};
  out.X.assign(ds.X.begin() + static_cast<long>(start) * ds.num_features,
               ds.X.begin() + static_cast<long>(start + count) * ds.num_features);
  out.y.assign(ds.y.begin() + start, ds.y.begin() + start + count);
  return out;
}

// Uneven micro-batches: 137, 89, 174 samples (sums to 400).
const std::vector<int> kMicrobatchSizes{137, 89, 174};

bool test_matches_full_batch(const Dataset &ds, const std::vector<float> &w) {
  auto full_grad = mse_gradient_sum(ds, w);
  std::vector<float> full_scaled(full_grad.size());
  for (size_t i = 0; i < full_grad.size(); ++i) full_scaled[i] = full_grad[i] / static_cast<float>(ds.num_samples);

  GradientAccumulator acc(w.size());
  int offset = 0;
  for (int sz : kMicrobatchSizes) {
    acc.accumulate(mse_gradient_sum(slice(ds, offset, sz), w), sz);
    offset += sz;
  }
  auto accumulated = acc.finalize_and_reset();

  bool ok = true;
  for (size_t i = 0; i < full_scaled.size(); ++i) {
    float rel = std::abs(full_scaled[i] - accumulated[i]) / std::max(1e-6f, std::abs(full_scaled[i]));
    if (rel > 1e-4f) ok = false;
  }
  std::printf("test 1 (correct scaling matches full-batch): %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool test_naive_scaling_is_wrong(const Dataset &ds, const std::vector<float> &w) {
  // Both scalings sum the SAME per-micro-batch gradients (associativity of
  // sum means this equals mse_gradient_sum(ds, w) directly) — the only
  // difference is what the sum gets divided by.
  auto raw_sum = mse_gradient_sum(ds, w);
  int num_microbatches = static_cast<int>(kMicrobatchSizes.size());

  std::vector<float> correct(raw_sum.size()), naive_wrong(raw_sum.size());
  for (size_t i = 0; i < raw_sum.size(); ++i) {
    correct[i] = raw_sum[i] / static_cast<float>(ds.num_samples);         // divide by total SAMPLES (400) — right
    naive_wrong[i] = raw_sum[i] / static_cast<float>(num_microbatches);   // divide by micro-batch COUNT (3) — wrong
  }

  float expected_ratio = static_cast<float>(ds.num_samples) / static_cast<float>(num_microbatches); // 133.3x
  float actual_ratio = naive_wrong[0] / correct[0];
  bool bug_is_real = std::abs(actual_ratio - expected_ratio) < 1.0f;
  std::printf("test 2 (naive scaling inflates the update %.1fx vs correct — expected %.1fx): %s\n", actual_ratio,
              expected_ratio, bug_is_real ? "PASS (bug reproduced as expected)" : "FAIL (bug not reproduced)");
  return bug_is_real;
}

bool test_composes_with_data_parallel() {
  constexpr int kNumFeatures = 8;
  constexpr int kNumSamples = 400;
  constexpr int kWorldSize = 4;
  constexpr int kSteps = 20;
  constexpr float kLr = 0.01f;
  constexpr uint16_t kBasePort = 35151;
  const int kShardSize = kNumSamples / kWorldSize; // 100/rank, split into 37+63 micro-batches

  std::vector<float> w_true(kNumFeatures);
  for (int j = 0; j < kNumFeatures; ++j) w_true[static_cast<size_t>(j)] = 0.5f + 0.1f * static_cast<float>(j);
  Dataset ds = make_synthetic_regression(kNumSamples, kNumFeatures, w_true, 0.05f, 42);
  std::vector<float> w0(kNumFeatures, 0.0f);

  std::vector<float> baseline_w = w0;
  std::vector<float> baseline_loss(kSteps);
  for (int step = 0; step < kSteps; ++step) {
    baseline_loss[static_cast<size_t>(step)] = mse_loss(ds, baseline_w);
    auto grad = mse_gradient_sum(ds, baseline_w);
    for (int j = 0; j < kNumFeatures; ++j) baseline_w[static_cast<size_t>(j)] -= kLr * grad[static_cast<size_t>(j)] / kNumSamples;
  }

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);

  std::vector<std::future<std::vector<float>>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() -> std::vector<float> {
      Dataset shard = slice(ds, r * kShardSize, kShardSize);
      Dataset mb1 = slice(shard, 0, 37);
      Dataset mb2 = slice(shard, 37, 63);

      std::vector<float> w = w0;
      std::vector<float> loss_curve(kSteps);
      for (int step = 0; step < kSteps; ++step) {
        loss_curve[static_cast<size_t>(step)] = mse_loss(ds, w);

        GradientAccumulator acc(w.size());
        acc.accumulate(mse_gradient_sum(mb1, w), 37);
        acc.accumulate(mse_gradient_sum(mb2, w), 63);
        auto mean = acc.finalize_and_reset(); // mean over this rank's 100 samples

        // ring_allreduce sums, not averages, so convert this rank's mean
        // back to a partial sum before all-reducing.
        std::vector<float> rank_grad(w.size());
        for (size_t i = 0; i < w.size(); ++i) rank_grad[i] = mean[i] * static_cast<float>(kShardSize);
        ring_allreduce(rank_grad.data(), rank_grad.size(), *ch);
        for (int j = 0; j < kNumFeatures; ++j) w[static_cast<size_t>(j)] -= kLr * rank_grad[static_cast<size_t>(j)] / kNumSamples;
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
  for (int step = 0; step < kSteps; ++step) {
    float rel = std::abs(baseline_loss[static_cast<size_t>(step)] - rank0_loss[static_cast<size_t>(step)]) /
                std::max(1e-6f, std::abs(baseline_loss[static_cast<size_t>(step)]));
    if (rel > 1e-2f) ok = false;
  }
  std::printf("test 3 (accumulation + data parallel matches baseline): %s (final loss %.6f vs %.6f)\n", ok ? "PASS" : "FAIL",
              rank0_loss.back(), baseline_loss.back());
  return ok;
}

} // namespace

int main() {
  constexpr int kNumFeatures = 8;
  std::vector<float> w_true(kNumFeatures);
  for (int j = 0; j < kNumFeatures; ++j) w_true[static_cast<size_t>(j)] = 0.5f + 0.1f * static_cast<float>(j);
  Dataset ds = make_synthetic_regression(400, kNumFeatures, w_true, 0.05f, 7);
  std::vector<float> w(kNumFeatures, 0.1f);

  bool ok = true;
  ok = test_matches_full_batch(ds, w) && ok;
  ok = test_naive_scaling_is_wrong(ds, w) && ok;
  ok = test_composes_with_data_parallel() && ok;

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
