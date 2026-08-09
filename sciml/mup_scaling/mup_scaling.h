//===- mup_scaling.h - muP output-layer LR scaling, tested directly -----===//
//
// PLAN.md Phase 18 step 9: Yang, G. & Hu, E.J. et al. (2021/2022),
// "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
// Hyperparameter Transfer" (muP) -- the entire claim under test:
// hyperparameters (specifically, the optimal learning rate) tuned at
// small width transfer to large width under the right parameterization,
// and do NOT transfer under standard parameterization (SP).
//
// Scope note: muP's full recipe is a table of init-variance and
// learning-rate multipliers per layer TYPE (input/hidden/output), derived
// from Tensor Programs' infinite-width limit theory. This step tests the
// single most commonly cited, practically load-bearing piece of that
// table for a simple 1-hidden-layer MLP: the OUTPUT (readout) layer's
// learning rate scaled down as 1/width, everything else (init variance,
// hidden-layer LR) left at standard fan-in scaling in both SP and muP --
// isolating the LR-transfer mechanism specifically, not reproducing the
// full multi-parameter table. A real, disclosed scope reduction, same
// pattern as every other "simplified vs. the full paper" note in this
// repo.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../ssm_layer/ssm_layer.h" // for Vec

#include <cmath>
#include <random>
#include <vector>

namespace sciml {

// y = W2 . tanh(W1*x + b1) + b2. x in R^d_in, hidden width n, scalar output.
struct MupMlpParams {
  int d_in, n;
  std::vector<double> W1, b1, W2; // sizes n*d_in, n, n
  double b2 = 0.0;

  int num_params() const { return n * d_in + n + n + 1; }

  static MupMlpParams random_init(int d_in, int n, uint64_t seed) {
    MupMlpParams p;
    p.d_in = d_in;
    p.n = n;
    std::mt19937_64 rng(seed);
    // Standard fan-in init variance for BOTH layers, same in SP and muP
    // here -- this step isolates the LEARNING-RATE-scaling mechanism,
    // not the init-scaling one (see file header).
    std::normal_distribution<double> d_in_dist(0.0, 1.0 / std::sqrt(static_cast<double>(d_in)));
    std::normal_distribution<double> hidden_dist(0.0, 1.0 / std::sqrt(static_cast<double>(n)));
    p.W1.resize(static_cast<size_t>(n * d_in));
    p.b1.assign(static_cast<size_t>(n), 0.0);
    p.W2.resize(static_cast<size_t>(n));
    for (double &v : p.W1) v = d_in_dist(rng);
    for (double &v : p.W2) v = hidden_dist(rng);
    p.b2 = 0.0;
    return p;
  }

  std::vector<double> flatten() const {
    std::vector<double> out;
    out.insert(out.end(), W1.begin(), W1.end());
    out.insert(out.end(), b1.begin(), b1.end());
    out.insert(out.end(), W2.begin(), W2.end());
    out.push_back(b2);
    return out;
  }
  static MupMlpParams from_flat(const std::vector<double> &flat, int d_in, int n) {
    MupMlpParams p;
    p.d_in = d_in;
    p.n = n;
    size_t off = 0;
    p.W1.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n * d_in)));
    off += static_cast<size_t>(n * d_in);
    p.b1.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n)));
    off += static_cast<size_t>(n);
    p.W2.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n)));
    off += static_cast<size_t>(n);
    p.b2 = flat[off];
    return p;
  }
};

inline double mup_mlp_forward(const MupMlpParams &p, const Vec &x) {
  double out = p.b2;
  for (int i = 0; i < p.n; ++i) {
    double s = p.b1[static_cast<size_t>(i)];
    for (int j = 0; j < p.d_in; ++j) s += p.W1[static_cast<size_t>(i * p.d_in + j)] * x[static_cast<size_t>(j)];
    out += p.W2[static_cast<size_t>(i)] * std::tanh(s);
  }
  return out;
}

struct RegressionSample {
  Vec x;
  double y;
};

inline double mse_loss(const std::vector<double> &flat, int d_in, int n, const std::vector<RegressionSample> &data) {
  MupMlpParams p = MupMlpParams::from_flat(flat, d_in, n);
  double total = 0.0;
  for (const auto &s : data) {
    double pred = mup_mlp_forward(p, s.x);
    double d = pred - s.y;
    total += d * d;
  }
  return total / static_cast<double>(data.size());
}

// One gradient-descent step with PER-PARAMETER-GROUP learning rates: the
// W1/b1 block gets `lr_input`, the W2/b2 (output/readout) block gets
// `lr_output`. Standard parametrization (SP) passes the SAME value for
// both; muP passes `lr_output = lr_input / n` (width-scaled down).
template <typename LossFn>
inline void mup_gd_step(std::vector<double> &flat, int d_in, int n, LossFn loss_fn, double lr_input,
                         double lr_output, double eps = 1e-4) {
  int input_block_size = n * d_in + n; // W1 + b1
  std::vector<double> grad(flat.size(), 0.0);
  for (size_t k = 0; k < flat.size(); ++k) {
    double orig = flat[k];
    flat[k] = orig + eps;
    double loss_plus = loss_fn(flat);
    flat[k] = orig - eps;
    double loss_minus = loss_fn(flat);
    flat[k] = orig;
    grad[k] = (loss_plus - loss_minus) / (2.0 * eps);
  }
  for (size_t k = 0; k < flat.size(); ++k) {
    double lr = (static_cast<int>(k) < input_block_size) ? lr_input : lr_output;
    flat[k] -= lr * grad[k];
  }
}

} // namespace sciml
