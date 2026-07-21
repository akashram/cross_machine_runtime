#pragma once

// Standard Adam optimizer over a flat float shard — not tied to any
// particular parameter shape, so it works identically whether it is
// updating a whole model (single-process baseline) or one rank's shard of
// it (ZeroStage1Optimizer).

#include <cmath>
#include <cstddef>
#include <vector>

namespace distributed_training {

struct AdamState {
  std::vector<float> m;
  std::vector<float> v;
  int t = 0;

  explicit AdamState(size_t n) : m(n, 0.0f), v(n, 0.0f) {}
};

inline void adam_step(std::vector<float> &param, const std::vector<float> &grad, AdamState &state, float lr,
                       float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
  state.t += 1;
  float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(state.t));
  float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(state.t));
  for (size_t i = 0; i < param.size(); ++i) {
    state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * grad[i];
    state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * grad[i] * grad[i];
    float mhat = state.m[i] / bias_correction1;
    float vhat = state.v[i] / bias_correction2;
    param[i] -= lr * mhat / (std::sqrt(vhat) + eps);
  }
}

} // namespace distributed_training
