// zero3_test.cpp — same baseline-vs-4-ranks structure as steps 7/8, now
// with parameters sharded too: each rank's ONLY persistent state across
// the whole run is a ZeroStage3Optimizer holding its 1/world_size shard.
// The full parameter vector is gathered fresh every epoch and never
// stored between epochs — the property being validated.
#include "zero3_optimizer.h"
#include "mlp.h"

#include <cmath>
#include <cstdio>
#include <future>
#include <random>
#include <vector>

using namespace distributed_training;

namespace {

struct ToyDataset {
  Matrix x;
  std::vector<int> labels;
  int total;
};

ToyDataset make_toy_dataset(std::mt19937 &rng) {
  constexpr int kClasses = 3;
  constexpr int kPerClass = 40;
  ToyDataset ds{Matrix(kClasses * kPerClass, 2), {}, kClasses * kPerClass};
  ds.labels.resize(static_cast<size_t>(ds.total));
  std::normal_distribution<float> noise(0.0f, 0.3f);
  const float centers[kClasses][2] = {{0.0f, 0.0f}, {3.0f, 3.0f}, {-3.0f, 3.0f}};
  int idx = 0;
  for (int c = 0; c < kClasses; ++c) {
    for (int s = 0; s < kPerClass; ++s) {
      ds.x(idx, 0) = centers[c][0] + noise(rng);
      ds.x(idx, 1) = centers[c][1] + noise(rng);
      ds.labels[static_cast<size_t>(idx)] = c;
      ++idx;
    }
  }
  return ds;
}

Matrix slice_rows(const Matrix &m, int start, int count) {
  Matrix out(count, m.cols());
  for (int i = 0; i < count; ++i)
    for (int j = 0; j < m.cols(); ++j) out(i, j) = m(start + i, j);
  return out;
}

float compute_loss(const MLP &mlp, const Matrix &x_data, const std::vector<int> &labels) {
  Tensor x(x_data);
  Tensor logits = mlp.forward(x);
  return softmax_cross_entropy(logits, labels).value()(0, 0);
}

} // namespace

int main() {
  constexpr int kEpochs = 60;
  constexpr float kLr = 0.05f;
  constexpr int kWorldSize = 4;
  constexpr uint16_t kBasePort = 35401;

  std::mt19937 data_rng(99);
  ToyDataset ds = make_toy_dataset(data_rng);
  int shard_size = ds.total / kWorldSize;

  std::mt19937 init_rng(42);
  MLP init_mlp({2, 8, 3}, init_rng);
  std::vector<float> initial_flat = flatten_params(init_mlp.parameters());

  MLP baseline_mlp({2, 8, 3}, init_rng);
  auto baseline_params = baseline_mlp.parameters();
  unflatten_params(baseline_params, initial_flat);
  AdamState baseline_state(total_param_count(baseline_params));

  std::vector<float> baseline_loss(kEpochs);
  for (int epoch = 0; epoch < kEpochs; ++epoch) {
    baseline_loss[static_cast<size_t>(epoch)] = compute_loss(baseline_mlp, ds.x, ds.labels);
    Tensor x(ds.x);
    Tensor logits = baseline_mlp.forward(x);
    Tensor loss = softmax_cross_entropy(logits, ds.labels);
    zero_grad(baseline_params);
    loss.backward();
    auto flat_params = flatten_params(baseline_params);
    auto flat_grad = flatten_grads(baseline_params);
    adam_step(flat_params, flat_grad, baseline_state, kLr);
    unflatten_params(baseline_params, flat_params);
  }

  auto channels = netcommon::make_tcp_loopback_mesh(kWorldSize, kBasePort);
  std::vector<std::future<std::vector<float>>> results;
  for (int r = 0; r < kWorldSize; ++r) {
    netcommon::Channel *ch = channels[static_cast<size_t>(r)].get();
    results.push_back(std::async(std::launch::async, [&, r, ch]() -> std::vector<float> {
      std::mt19937 rank_rng(1000 + r);
      MLP rank_mlp({2, 8, 3}, rank_rng);
      auto rank_params = rank_mlp.parameters();
      size_t total = total_param_count(rank_params);

      ZeroStage3Optimizer opt(total, r, kWorldSize, kLr);
      std::vector<float> seed_full = initial_flat;
      seed_full.resize(opt.padded_total_params(), 0.0f);
      opt.init_from_full(seed_full);
      // From here on, opt.params_shard_ (private) is this rank's ENTIRE
      // persistent parameter memory — seed_full and initial_flat are never
      // touched again.

      Matrix shard_x = slice_rows(ds.x, r * shard_size, shard_size);
      std::vector<int> shard_labels(ds.labels.begin() + r * shard_size, ds.labels.begin() + (r + 1) * shard_size);

      std::vector<float> loss_curve(kEpochs);
      for (int epoch = 0; epoch < kEpochs; ++epoch) {
        std::vector<float> full_params = opt.gather_full_params(*ch); // transient — discarded at end of scope
        unflatten_params(rank_params, full_params);

        if (r == 0) loss_curve[static_cast<size_t>(epoch)] = compute_loss(rank_mlp, ds.x, ds.labels);

        Tensor x(shard_x);
        Tensor logits = rank_mlp.forward(x);
        Tensor loss = softmax_cross_entropy(logits, shard_labels);
        zero_grad(rank_params);
        loss.backward();

        auto local_grad = flatten_grads(rank_params);
        local_grad.resize(opt.padded_total_params(), 0.0f);
        for (float &g : local_grad) g /= static_cast<float>(kWorldSize);

        opt.step(local_grad, *ch);
      }
      return r == 0 ? loss_curve : std::vector<float>{};
    }));
  }

  std::vector<float> zero3_loss;
  for (auto &f : results) {
    auto lc = f.get();
    if (!lc.empty()) zero3_loss = std::move(lc);
  }

  bool ok = true;
  std::printf("epoch  baseline_loss  zero3_loss  rel_diff\n");
  for (int epoch = 0; epoch < kEpochs; epoch += 10) {
    float b = baseline_loss[static_cast<size_t>(epoch)];
    float z = zero3_loss[static_cast<size_t>(epoch)];
    float rel_diff = std::abs(b - z) / std::max(1e-3f, std::abs(b));
    std::printf("%5d  %13.6f  %10.6f  %8.4f\n", epoch, b, z, rel_diff);
    if (rel_diff > 0.15f) ok = false;
  }
  std::printf("final: baseline=%.6f zero3=%.6f\n", baseline_loss.back(), zero3_loss.back());
  if (baseline_loss.back() > baseline_loss.front() * 0.3f || zero3_loss.back() > zero3_loss.front() * 0.3f) {
    std::printf("one or both did not converge meaningfully\n");
    ok = false;
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
