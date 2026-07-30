#pragma once

// Shared fixture for Phase 14: a toy classification task built on
// distributed_training::MLP + the generic reverse-mode tape
// (distributed_training/autograd/autograd.h) -- the SAME model class
// distributed_training/full_training_loop already trains, reused here
// rather than reinvented so Phase 14's model is directly comparable to
// Phase 6's. 3-class Gaussian blobs in 2D: continuous, unconstrained
// features are exactly FGSM/PGD's classic setting (Goodfellow et al.
// 2014's original demonstration was on continuous pixel inputs, not
// discrete tokens), unlike attacking a language model's discrete token
// ids, which would need a much larger scope (attacking the CONTINUOUS
// embedding lookup instead, a real but separate research question this
// phase does not take on).

#include "../../distributed_training/autograd/mlp.h"

#include <random>
#include <vector>

namespace adversarial {

using distributed_training::Matrix;
using distributed_training::MLP;
using distributed_training::Tensor;
using distributed_training::softmax_cross_entropy;
using distributed_training::sgd_step;
using distributed_training::zero_grad;

struct Dataset {
  Matrix x;
  std::vector<int> labels;
};

// `centers`: one [x,y] pair per class. Points are centers[c] + Gaussian
// noise (stddev `noise_stddev`), `per_class` points per class.
inline Dataset make_blob_dataset(const std::vector<std::pair<float, float>> &centers, int per_class,
                                  float noise_stddev, std::mt19937 &rng) {
  int classes = static_cast<int>(centers.size());
  Dataset ds;
  ds.x = Matrix(classes * per_class, 2);
  ds.labels.resize(static_cast<std::size_t>(classes * per_class));
  std::normal_distribution<float> noise(0.0f, noise_stddev);
  int idx = 0;
  for (int c = 0; c < classes; ++c) {
    for (int s = 0; s < per_class; ++s) {
      ds.x(idx, 0) = centers[static_cast<std::size_t>(c)].first + noise(rng);
      ds.x(idx, 1) = centers[static_cast<std::size_t>(c)].second + noise(rng);
      ds.labels[static_cast<std::size_t>(idx)] = c;
      ++idx;
    }
  }
  return ds;
}

// The standard 3-blob task shared across every Phase 14 step needing one
// -- same centers/spread every time so results are comparable across
// steps, same convention as rag/corpus.h being shared across Phase 13
// steps 4/6/7/8.
inline Dataset make_standard_dataset(int per_class, std::mt19937 &rng) {
  return make_blob_dataset({{0.0f, 0.0f}, {4.0f, 4.0f}, {-4.0f, 4.0f}}, per_class, 0.6f, rng);
}

inline MLP train_classifier(const Dataset &train, const std::vector<int> &layer_dims, int epochs, float lr,
                             std::mt19937 &rng) {
  MLP mlp(layer_dims, rng);
  auto params = mlp.parameters();
  for (int epoch = 0; epoch < epochs; ++epoch) {
    zero_grad(params);
    Tensor x(train.x);
    Tensor logits = mlp.forward(x);
    Tensor loss = softmax_cross_entropy(logits, train.labels);
    loss.backward();
    sgd_step(params, lr);
  }
  return mlp;
}

inline int argmax_row(const Matrix &m, int row) {
  int best = 0;
  float best_v = m(row, 0);
  for (int c = 1; c < m.cols(); ++c)
    if (m(row, c) > best_v) { best_v = m(row, c); best = c; }
  return best;
}

inline float accuracy(const MLP &mlp, const Matrix &x, const std::vector<int> &labels) {
  Tensor xt(x);
  Tensor logits = mlp.forward(xt);
  const Matrix &z = logits.value();
  int correct = 0;
  for (int i = 0; i < z.rows(); ++i)
    if (argmax_row(z, i) == labels[static_cast<std::size_t>(i)]) ++correct;
  return static_cast<float>(correct) / static_cast<float>(z.rows());
}

} // namespace adversarial
