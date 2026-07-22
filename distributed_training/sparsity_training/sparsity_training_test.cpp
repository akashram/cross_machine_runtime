// sparsity_training_test.cpp — trains a toy MLP densely for a warmup
// period, prunes the first layer's weight to 2:4 (NVIDIA ASP recipe:
// prune once, hold the mask fixed), then fine-tunes with the mask
// re-applied to the gradient every step so pruned entries never drift
// from exactly zero. Validates: (1) the 2:4 property genuinely holds
// throughout fine-tuning, not just right after pruning, (2) the
// sparsified model still converges to comparable accuracy vs. a fully
// dense baseline trained for the same total steps.
#include "sparsity_2_4.h"
#include "mlp.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace distributed_training;

namespace {

struct ToyDataset {
  Matrix x;
  std::vector<int> labels;
  int total;
};

// 4D input (in_dim must be a multiple of 4 for 2:4 sparsity along the K
// axis of the first layer's weight) — 3 well-separated clusters.
ToyDataset make_toy_dataset(std::mt19937 &rng) {
  constexpr int kClasses = 3;
  constexpr int kPerClass = 40;
  ToyDataset ds{Matrix(kClasses * kPerClass, 4), {}, kClasses * kPerClass};
  ds.labels.resize(static_cast<size_t>(ds.total));
  std::normal_distribution<float> noise(0.0f, 0.3f);
  const float centers[kClasses][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {3.0f, 3.0f, 3.0f, 3.0f}, {-3.0f, 3.0f, -3.0f, 3.0f}};
  int idx = 0;
  for (int c = 0; c < kClasses; ++c) {
    for (int s = 0; s < kPerClass; ++s) {
      for (int j = 0; j < 4; ++j) ds.x(idx, j) = centers[c][j] + noise(rng);
      ds.labels[static_cast<size_t>(idx)] = c;
      ++idx;
    }
  }
  return ds;
}

float accuracy(const MLP &mlp, const Matrix &x, const std::vector<int> &labels) {
  Tensor xt(x);
  Matrix logits = mlp.forward(xt).value();
  int correct = 0;
  for (int i = 0; i < logits.rows(); ++i) {
    int argmax = 0;
    float best = logits(i, 0);
    for (int j = 1; j < logits.cols(); ++j)
      if (logits(i, j) > best) { best = logits(i, j); argmax = j; }
    if (argmax == labels[static_cast<size_t>(i)]) ++correct;
  }
  return static_cast<float>(correct) / static_cast<float>(logits.rows());
}

float loss_of(const MLP &mlp, const Matrix &x, const std::vector<int> &labels) {
  Tensor xt(x);
  return softmax_cross_entropy(mlp.forward(xt), labels).value()(0, 0);
}

} // namespace

int main() {
  constexpr int kWarmupEpochs = 80;
  constexpr int kFinetuneEpochs = 120;
  constexpr float kLr = 0.1f;
  std::mt19937 data_rng(15);
  ToyDataset ds = make_toy_dataset(data_rng);

  std::mt19937 init_rng(3);
  MLP mlp({4, 16, 3}, init_rng);
  auto params = mlp.parameters(); // [weight0, bias0, weight1, bias1]

  // --- Dense warmup.
  for (int epoch = 0; epoch < kWarmupEpochs; ++epoch) {
    Tensor x(ds.x);
    Tensor loss = softmax_cross_entropy(mlp.forward(x), ds.labels);
    zero_grad(params);
    loss.backward();
    sgd_step(params, kLr);
  }
  float dense_loss = loss_of(mlp, ds.x, ds.labels);
  float dense_acc = accuracy(mlp, ds.x, ds.labels);
  std::printf("after dense warmup (%d epochs): loss=%.4f accuracy=%.1f%%\n", kWarmupEpochs, dense_loss, dense_acc * 100.0f);

  // --- Prune layer 0's weight to 2:4, once, mask held fixed from here on.
  Tensor weight0 = params[0]; // [4 x 16]
  Matrix mask = compute_2_4_mask(weight0.value());
  Matrix pruned_weight = weight0.value();
  apply_mask_inplace(pruned_weight, mask);
  weight0.mutable_value() = pruned_weight;

  bool mask_holds_immediately = verify_2_4_property(weight0.value());
  std::printf("2:4 property immediately after pruning: %s\n", mask_holds_immediately ? "PASS" : "FAIL");

  std::vector<Tensor> other_params{params[1], params[2], params[3]}; // bias0, weight1, bias1 -- trained normally

  // --- Sparse fine-tune: weight0 updated only through the masked gradient.
  for (int epoch = 0; epoch < kFinetuneEpochs; ++epoch) {
    Tensor x(ds.x);
    Tensor loss = softmax_cross_entropy(mlp.forward(x), ds.labels);
    zero_grad(params);
    loss.backward();

    Matrix masked_grad = weight0.grad();
    apply_mask_inplace(masked_grad, mask);
    weight0.mutable_value().add_inplace(masked_grad, -kLr);

    sgd_step(other_params, kLr);
  }

  bool mask_holds_after_finetune = verify_2_4_property(weight0.value());
  float sparse_loss = loss_of(mlp, ds.x, ds.labels);
  float sparse_acc = accuracy(mlp, ds.x, ds.labels);
  std::printf("after sparse fine-tune (%d epochs): loss=%.4f accuracy=%.1f%%\n", kFinetuneEpochs, sparse_loss,
              sparse_acc * 100.0f);
  std::printf("2:4 property still holds after fine-tuning: %s\n", mask_holds_after_finetune ? "PASS" : "FAIL");

  bool ok = mask_holds_immediately && mask_holds_after_finetune && sparse_acc >= 0.85f && sparse_loss < dense_loss * 5.0f;
  std::printf("sparsified model still learns well (accuracy >= 85%%): %s\n", sparse_acc >= 0.85f ? "PASS" : "FAIL");

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
