//===- ebm.h - energy-based model via contrastive divergence ------------===//
//
// PLAN.md Phase 18 step 8: LeCun, Chopra, Hadsell, Ranzato & Huang (2006),
// "A Tutorial on Energy-Based Learning" -- a scalar energy function
// E_theta(x) with implied density p(x) ~ exp(-E_theta(x)), trained via
// contrastive divergence (CD) with short-run Langevin MCMC negative
// sampling (Hinton 2002's CD training procedure; the Langevin sampler
// itself follows the short-run-MCMC EBM training recipe popularized by
// Du & Mordatch 2019), on the SAME toy two-cluster 2D target distribution
// step 7's diffusion model used -- a direct architecture-family
// comparison, per PLAN.md's explicit ask for this step.
//
// CD training, correctly done: for one training step, generate negative
// samples x_neg via Langevin MCMC using the CURRENT theta, then treat
// x_neg as FIXED (stop-gradient through the sampling chain -- the
// standard CD convention) when computing the parameter update. The
// surrogate loss `L(theta) = mean(E_theta(x_pos)) - mean(E_theta(x_neg))`,
// finite-differenced w.r.t. theta with x_pos/x_neg held fixed, gives
// EXACTLY the contrastive divergence gradient (push energy down at real
// data, up at model samples) -- not an approximation of it, because only
// E_theta's direct dependence on theta is being differentiated, matching
// what "stop-gradient through the MCMC chain" means in the real
// algorithm. Reuses `sciml::finite_diff_gd_step` from step 6 directly.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../diffusion/diffusion.h" // for Point2D, sample_target, kClusterA/B/Std
#include "../ssm_layer/ssm_layer.h" // for finite_diff_gd_step

#include <cmath>
#include <random>

namespace sciml {

// E_theta(x) = w2 . tanh(W1*x + b1) + b2. Scalar energy output.
struct EBMParams {
  int h = 8;
  std::vector<double> W1, b1, w2; // sizes h*2, h, h
  double b2 = 0.0;

  int num_params() const { return h * 2 + h + h + 1; }

  static EBMParams random_init(int h, uint64_t seed) {
    EBMParams p;
    p.h = h;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> d(0.0, 0.5);
    p.W1.resize(static_cast<size_t>(h * 2));
    p.b1.resize(static_cast<size_t>(h));
    p.w2.resize(static_cast<size_t>(h));
    for (double &v : p.W1) v = d(rng);
    for (double &v : p.b1) v = d(rng);
    for (double &v : p.w2) v = d(rng);
    p.b2 = d(rng);
    return p;
  }

  std::vector<double> flatten() const {
    std::vector<double> out;
    out.insert(out.end(), W1.begin(), W1.end());
    out.insert(out.end(), b1.begin(), b1.end());
    out.insert(out.end(), w2.begin(), w2.end());
    out.push_back(b2);
    return out;
  }
  static EBMParams from_flat(const std::vector<double> &flat, int h) {
    EBMParams p;
    p.h = h;
    size_t off = 0;
    p.W1.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(h * 2)));
    off += static_cast<size_t>(h * 2);
    p.b1.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(h)));
    off += static_cast<size_t>(h);
    p.w2.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(h)));
    off += static_cast<size_t>(h);
    p.b2 = flat[off];
    return p;
  }
};

inline double energy(const EBMParams &p, const Point2D &x) {
  double e = p.b2;
  for (int i = 0; i < p.h; ++i) {
    double s = p.b1[static_cast<size_t>(i)];
    s += p.W1[static_cast<size_t>(i * 2 + 0)] * x[0];
    s += p.W1[static_cast<size_t>(i * 2 + 1)] * x[1];
    e += p.w2[static_cast<size_t>(i)] * std::tanh(s);
  }
  return e;
}

// Numerical gradient of E w.r.t. x (2D), central differences -- the
// Langevin sampler's driving force, same finite-difference convention as
// every other gradient in this phase.
inline Point2D energy_grad_x(const EBMParams &p, const Point2D &x, double eps = 1e-5) {
  Point2D x_plus0 = x, x_minus0 = x;
  x_plus0[0] += eps; x_minus0[0] -= eps;
  double gx = (energy(p, x_plus0) - energy(p, x_minus0)) / (2.0 * eps);
  Point2D x_plus1 = x, x_minus1 = x;
  x_plus1[1] += eps; x_minus1[1] -= eps;
  double gy = (energy(p, x_plus1) - energy(p, x_minus1)) / (2.0 * eps);
  return {gx, gy};
}

// Short-run Langevin MCMC: x_{k+1} = x_k - (step/2)*dE/dx + sqrt(step)*noise.
// Descends the energy landscape (toward high-density / low-energy
// regions) with injected noise, the standard EBM sampling recipe.
inline Point2D langevin_sample(const EBMParams &p, Point2D x, int num_steps, double step_size, std::mt19937 &rng) {
  std::normal_distribution<double> noise(0.0, 1.0);
  for (int k = 0; k < num_steps; ++k) {
    Point2D g = energy_grad_x(p, x);
    x[0] = x[0] - (step_size / 2.0) * g[0] + std::sqrt(step_size) * noise(rng);
    x[1] = x[1] - (step_size / 2.0) * g[1] + std::sqrt(step_size) * noise(rng);
  }
  return x;
}

// Contrastive-divergence surrogate loss for ONE training step:
// mean(E(x_pos)) - mean(E(x_neg)), with x_pos/x_neg held FIXED (captured
// by value) -- finite-differencing this w.r.t. theta gives exactly the CD
// gradient, since the sampling chain that produced x_neg is not
// re-differentiated (the correct "stop-gradient through MCMC" behavior).
inline double cd_loss(const std::vector<double> &flat, int h, const std::vector<Point2D> &x_pos,
                       const std::vector<Point2D> &x_neg) {
  EBMParams p = EBMParams::from_flat(flat, h);
  double e_pos = 0.0, e_neg = 0.0;
  for (const auto &x : x_pos) e_pos += energy(p, x);
  for (const auto &x : x_neg) e_neg += energy(p, x);
  return e_pos / static_cast<double>(x_pos.size()) - e_neg / static_cast<double>(x_neg.size());
}

} // namespace sciml
