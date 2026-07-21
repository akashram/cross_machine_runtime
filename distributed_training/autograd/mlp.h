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

// Flatten/unflatten a parameter list to/from one contiguous vector<float>.
// Steps 7+ (ZeRO, tensor/pipeline/3D parallelism, checkpointing) shard,
// communicate, and reassemble parameters and gradients as flat float
// buffers (matching networking/ring_allreduce's and collectives'
// `float*`-based interface), not as a list of differently-shaped Tensors
// — these are the glue between the two.
inline std::vector<float> flatten_params(const std::vector<Tensor> &params) {
  std::vector<float> flat;
  for (const auto &p : params) {
    const Matrix &m = p.value();
    for (int r = 0; r < m.rows(); ++r)
      for (int c = 0; c < m.cols(); ++c) flat.push_back(m(r, c));
  }
  return flat;
}

inline std::vector<float> flatten_grads(const std::vector<Tensor> &params) {
  std::vector<float> flat;
  for (const auto &p : params) {
    const Matrix &g = p.grad();
    for (int r = 0; r < g.rows(); ++r)
      for (int c = 0; c < g.cols(); ++c) flat.push_back(g(r, c));
  }
  return flat;
}

// Copies a flat buffer (as produced by flatten_params) back into the
// parameters' value Matrices. `flat` may be longer than needed (e.g.
// padded to a multiple of world_size for uniform sharding) — only the
// first total_param_count(params) entries are read.
inline void unflatten_params(const std::vector<Tensor> &params, const std::vector<float> &flat) {
  size_t idx = 0;
  for (const auto &p : params) {
    Matrix &m = p.mutable_value();
    for (int r = 0; r < m.rows(); ++r)
      for (int c = 0; c < m.cols(); ++c) m(r, c) = flat[idx++];
  }
}

inline size_t total_param_count(const std::vector<Tensor> &params) {
  size_t n = 0;
  for (const auto &p : params) n += p.value().size();
  return n;
}

} // namespace distributed_training
