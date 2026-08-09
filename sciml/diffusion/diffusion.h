//===- diffusion.h - DDPM denoising diffusion on a toy 2D distribution ---===//
//
// PLAN.md Phase 18 step 7: Sohl-Dickstein et al. (2015) and Ho, Jain &
// Abbeel (2020) DDPM -- a small noise-prediction network trained to
// reverse a fixed Gaussian forward-noising process, sampled by iterative
// ancestral denoising, on a toy 2D two-cluster target distribution.
//
// Scope note: this step implements DDPM only, not flow matching (Lipman
// et al. 2023) -- PLAN.md phrases step 7 as "diffusion and/or flow
// matching," and DDPM alone is enough real, trainable, sampleable
// machinery to demonstrate the family; flow matching is left as a real,
// disclosed scope reduction rather than a shallow second implementation.
//
// Training reuses `sciml::finite_diff_gd_step` from step 6's
// `ssm_layer.h` directly (the same generic trainer, not a third copy) --
// the noise-prediction network's loss is a plain MSE over randomly
// sampled (x0, t, noise) triples, small enough in parameter count for
// finite-difference gradients to be tractable at this toy scale.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../ssm_layer/ssm_layer.h" // for finite_diff_gd_step

#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace sciml {

using Point2D = std::array<double, 2>;

// ---------------------------------------------------------------------
// Toy target distribution: a 50/50 mixture of two well-separated 2D
// Gaussian clusters, centered at (-2,0) and (2,0), std 0.3 each.
// ---------------------------------------------------------------------
inline const Point2D kClusterA = {-2.0, 0.0};
inline const Point2D kClusterB = {2.0, 0.0};
inline constexpr double kClusterStd = 0.3;

inline Point2D sample_target(std::mt19937 &rng) {
  std::bernoulli_distribution which(0.5);
  std::normal_distribution<double> noise(0.0, kClusterStd);
  const Point2D &center = which(rng) ? kClusterA : kClusterB;
  return {center[0] + noise(rng), center[1] + noise(rng)};
}

// ---------------------------------------------------------------------
// Linear beta schedule (Ho et al. 2020's original choice, scaled to a
// smaller T for this toy setting) and the derived alpha/alpha_bar arrays.
// ---------------------------------------------------------------------
struct NoiseSchedule {
  int T;
  std::vector<double> beta, alpha, alpha_bar;

  static NoiseSchedule linear(int T, double beta_start = 1e-4, double beta_end = 0.02) {
    NoiseSchedule s;
    s.T = T;
    s.beta.resize(static_cast<size_t>(T));
    s.alpha.resize(static_cast<size_t>(T));
    s.alpha_bar.resize(static_cast<size_t>(T));
    double running = 1.0;
    for (int t = 0; t < T; ++t) {
      double frac = T > 1 ? static_cast<double>(t) / static_cast<double>(T - 1) : 0.0;
      s.beta[static_cast<size_t>(t)] = beta_start + frac * (beta_end - beta_start);
      s.alpha[static_cast<size_t>(t)] = 1.0 - s.beta[static_cast<size_t>(t)];
      running *= s.alpha[static_cast<size_t>(t)];
      s.alpha_bar[static_cast<size_t>(t)] = running;
    }
    return s;
  }
};

// Forward diffusion: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps.
inline Point2D q_sample(const Point2D &x0, int t, const Point2D &eps, const NoiseSchedule &sched) {
  double a_bar = sched.alpha_bar[static_cast<size_t>(t)];
  double sqrt_a = std::sqrt(a_bar), sqrt_1ma = std::sqrt(1.0 - a_bar);
  return {sqrt_a * x0[0] + sqrt_1ma * eps[0], sqrt_a * x0[1] + sqrt_1ma * eps[1]};
}

// ---------------------------------------------------------------------
// Noise-prediction network: eps_theta(x, t_normalized) = W2*tanh(W1*[x0,x1,t]+b1)+b2.
// Input dim 3 (2D point + normalized timestep), hidden width h, output dim 2.
// ---------------------------------------------------------------------
struct DiffusionNetParams {
  int h = 8;
  std::vector<double> W1, b1, W2, b2; // sizes h*3, h, 2*h, 2

  int num_params() const { return h * 3 + h + 2 * h + 2; }

  static DiffusionNetParams random_init(int h, uint64_t seed) {
    DiffusionNetParams p;
    p.h = h;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> d(0.0, 0.5);
    p.W1.resize(static_cast<size_t>(h * 3));
    p.b1.resize(static_cast<size_t>(h));
    p.W2.resize(static_cast<size_t>(2 * h));
    p.b2.resize(2);
    for (double &v : p.W1) v = d(rng);
    for (double &v : p.b1) v = d(rng);
    for (double &v : p.W2) v = d(rng);
    for (double &v : p.b2) v = d(rng);
    return p;
  }

  std::vector<double> flatten() const {
    std::vector<double> out;
    out.insert(out.end(), W1.begin(), W1.end());
    out.insert(out.end(), b1.begin(), b1.end());
    out.insert(out.end(), W2.begin(), W2.end());
    out.insert(out.end(), b2.begin(), b2.end());
    return out;
  }
  static DiffusionNetParams from_flat(const std::vector<double> &flat, int h) {
    DiffusionNetParams p;
    p.h = h;
    size_t off = 0;
    p.W1.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(h * 3)));
    off += static_cast<size_t>(h * 3);
    p.b1.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(h)));
    off += static_cast<size_t>(h);
    p.W2.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(2 * h)));
    off += static_cast<size_t>(2 * h);
    p.b2.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + 2));
    return p;
  }
};

inline Point2D eps_theta(const DiffusionNetParams &p, const Point2D &x, double t_norm) {
  std::vector<double> in = {x[0], x[1], t_norm};
  std::vector<double> hid(static_cast<size_t>(p.h));
  for (int i = 0; i < p.h; ++i) {
    double s = p.b1[static_cast<size_t>(i)];
    for (int j = 0; j < 3; ++j) s += p.W1[static_cast<size_t>(i * 3 + j)] * in[static_cast<size_t>(j)];
    hid[static_cast<size_t>(i)] = std::tanh(s);
  }
  Point2D out{};
  for (int i = 0; i < 2; ++i) {
    double s = p.b2[static_cast<size_t>(i)];
    for (int j = 0; j < p.h; ++j) s += p.W2[static_cast<size_t>(i * p.h + j)] * hid[static_cast<size_t>(j)];
    out[static_cast<size_t>(i)] = s;
  }
  return out;
}

// DDPM training loss: MSE between predicted and actual noise, averaged
// over a fixed batch of (x0, t, eps) triples (pre-sampled once so the
// finite-difference gradient sees a fixed loss landscape per step, not a
// re-randomized one for every perturbation).
struct TrainingSample {
  Point2D x0, eps;
  int t;
};

inline double diffusion_loss(const std::vector<double> &flat, int h, const NoiseSchedule &sched,
                              const std::vector<TrainingSample> &batch) {
  DiffusionNetParams p = DiffusionNetParams::from_flat(flat, h);
  double total = 0.0;
  for (const auto &s : batch) {
    Point2D x_t = q_sample(s.x0, s.t, s.eps, sched);
    double t_norm = static_cast<double>(s.t) / static_cast<double>(sched.T - 1);
    Point2D pred = eps_theta(p, x_t, t_norm);
    double d0 = pred[0] - s.eps[0], d1 = pred[1] - s.eps[1];
    total += d0 * d0 + d1 * d1;
  }
  return total / static_cast<double>(batch.size());
}

// Ancestral DDPM sampling: x_T ~ N(0,I), iteratively denoise down to x_0.
inline Point2D ddpm_sample(const DiffusionNetParams &p, const NoiseSchedule &sched, std::mt19937 &rng) {
  std::normal_distribution<double> std_normal(0.0, 1.0);
  Point2D x = {std_normal(rng), std_normal(rng)};
  for (int t = sched.T - 1; t >= 0; --t) {
    double t_norm = static_cast<double>(t) / static_cast<double>(sched.T - 1);
    Point2D predicted_eps = eps_theta(p, x, t_norm);
    double alpha_t = sched.alpha[static_cast<size_t>(t)];
    double alpha_bar_t = sched.alpha_bar[static_cast<size_t>(t)];
    double beta_t = sched.beta[static_cast<size_t>(t)];
    double coef = beta_t / std::sqrt(1.0 - alpha_bar_t);
    Point2D mean = {(x[0] - coef * predicted_eps[0]) / std::sqrt(alpha_t),
                     (x[1] - coef * predicted_eps[1]) / std::sqrt(alpha_t)};
    if (t > 0) {
      double sigma_t = std::sqrt(beta_t);
      x = {mean[0] + sigma_t * std_normal(rng), mean[1] + sigma_t * std_normal(rng)};
    } else {
      x = mean;
    }
  }
  return x;
}

} // namespace sciml
