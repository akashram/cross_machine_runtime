#pragma once

// A tiny MLP built on autograd.h — the toy model shared by steps 7-25
// (ZeRO, tensor/pipeline/3D parallelism, MoE, SyncBatchNorm, full training
// loop, SFT/reward model/PPO/DPO). Scaled for CPU correctness validation
// (tens to low-hundreds of parameters), not for representing a real LLM —
// see distributed_training/README.md and each step's own README for why
// that is the right tradeoff without GPU hardware.

#include <random>
#include <vector>

#include "autograd.h"

namespace distributed_training {

struct Linear {
  Tensor weight; // [in_dim x out_dim]
  Tensor bias;   // [1 x out_dim]

  Linear(int in_dim, int out_dim, std::mt19937 &rng) {
    float stddev = std::sqrt(2.0f / static_cast<float>(in_dim)); // He init, since ReLU follows in MLP
    weight = Tensor(Matrix::random(in_dim, out_dim, rng, stddev));
    bias = Tensor(Matrix::zeros(1, out_dim));
  }

  Tensor forward(const Tensor &x) const { return add_bias(matmul(x, weight), bias); }
};

class MLP {
public:
  // `layer_dims` = [input_dim, hidden1, hidden2, ..., output_dim]. ReLU
  // between every layer except the last (logits stay linear — softmax is
  // applied inside the loss, not the model).
  MLP(const std::vector<int> &layer_dims, std::mt19937 &rng) {
    for (size_t i = 0; i + 1 < layer_dims.size(); ++i) {
      layers_.emplace_back(layer_dims[i], layer_dims[i + 1], rng);
    }
  }

  Tensor forward(const Tensor &x) const {
    Tensor h = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
      h = layers_[i].forward(h);
      if (i + 1 < layers_.size()) h = relu(h);
    }
    return h;
  }

  std::vector<Tensor> parameters() const {
    std::vector<Tensor> params;
    for (auto &l : layers_) {
      params.push_back(l.weight);
      params.push_back(l.bias);
    }
    return params;
  }

  const std::vector<Linear> &layers() const { return layers_; }

private:
  std::vector<Linear> layers_;
};

inline void zero_grad(const std::vector<Tensor> &params) {
  for (auto &p : params) p.zero_grad();
}

inline void sgd_step(const std::vector<Tensor> &params, float lr) {
  for (auto &p : params) p.mutable_value().add_inplace(p.grad(), -lr);
}

} // namespace distributed_training
