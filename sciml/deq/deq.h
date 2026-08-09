//===- deq.h - Deep Equilibrium Model, implicit-function-theorem grad ---===//
//
// PLAN.md Phase 18 step 5: Bai, Kolter & Koltun (2019), "Deep Equilibrium
// Models" -- an implicit-depth layer defined by a fixed point
// z* = f_theta(z*, x), found by plain fixed-point iteration (Broyden's
// method, the paper's actual choice, is noted as a possible upgrade but
// not required for this step -- fixed-point iteration converges fine for
// the contraction-mapping-shaped f used here), with backprop through the
// fixed point via the IMPLICIT FUNCTION THEOREM instead of unrolling the
// iteration.
//
// Implicit function theorem derivation: differentiating z* = f(z*,x;theta)
// w.r.t. theta gives dz*/dtheta = (df/dz)*(dz*/dtheta) + df/dtheta, so
//   (I - df/dz) * dz*/dtheta = df/dtheta
//   dz*/dtheta = (I - df/dz)^-1 * df/dtheta
// Then dL/dtheta = (dL/dz*) * dz*/dtheta = g^T (I-Jz)^-1 Jtheta, where
// g = dL/dz*. Solving v from (I - Jz)^T v = g (reusing
// `sciml::detail::solve_linear` from ode_solver.h -- the same
// Gaussian-elimination solve step 1's backward_euler uses for its Newton
// steps) gives dL/dtheta = v^T * Jtheta directly, WITHOUT ever
// differentiating through the fixed-point iteration's individual steps --
// the whole point of the implicit function theorem approach: O(1) memory
// in the number of iterations, same shape as step 4's adjoint method being
// O(1) memory in the number of solver steps.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../ode_solver/ode_solver.h" // for sciml::detail::solve_linear

#include <cmath>
#include <random>

namespace sciml {

// f_theta(z, x) = tanh(Wz*z + Wx*x + b). z, x both in R^n (same dimension,
// so z is a valid candidate for a fixed point given a fixed x).
struct DEQParams {
  int n = 3;
  std::vector<double> Wz, Wx, b; // sizes n*n, n*n, n

  int num_params() const { return n * n + n * n + n; }

  static DEQParams random_init(int n, uint64_t seed) {
    DEQParams p;
    p.n = n;
    std::mt19937_64 rng(seed);
    // Scale weights down so f is a contraction (tanh already bounds the
    // output; keeping Wz's spectral radius comfortably below 1 in
    // expectation is what makes fixed-point ITERATION (not just the
    // fixed point's existence) converge reliably).
    std::normal_distribution<double> d(0.0, 0.3 / std::sqrt(static_cast<double>(n)));
    p.Wz.resize(static_cast<size_t>(n * n));
    p.Wx.resize(static_cast<size_t>(n * n));
    p.b.resize(static_cast<size_t>(n));
    for (double &v : p.Wz) v = d(rng);
    for (double &v : p.Wx) v = d(rng);
    for (double &v : p.b) v = d(rng);
    return p;
  }

  std::vector<double> flatten() const {
    std::vector<double> out;
    out.insert(out.end(), Wz.begin(), Wz.end());
    out.insert(out.end(), Wx.begin(), Wx.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
  }

  static DEQParams from_flat(const std::vector<double> &flat, int n) {
    DEQParams p;
    p.n = n;
    size_t off = 0;
    p.Wz.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n * n)));
    off += static_cast<size_t>(n * n);
    p.Wx.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n * n)));
    off += static_cast<size_t>(n * n);
    p.b.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n)));
    return p;
  }
};

inline State deq_f(const DEQParams &p, const State &z, const State &x) {
  State out(static_cast<size_t>(p.n), 0.0);
  for (int i = 0; i < p.n; ++i) {
    double s = p.b[static_cast<size_t>(i)];
    for (int j = 0; j < p.n; ++j) {
      s += p.Wz[static_cast<size_t>(i * p.n + j)] * z[static_cast<size_t>(j)];
      s += p.Wx[static_cast<size_t>(i * p.n + j)] * x[static_cast<size_t>(j)];
    }
    out[static_cast<size_t>(i)] = std::tanh(s);
  }
  return out;
}

struct FixedPointResult {
  State z_star;
  int iterations;
  bool converged;
};

// Plain fixed-point (Picard) iteration: z_{k+1} = f(z_k, x). Converges
// when f is a contraction near z* -- `DEQParams::random_init`'s weight
// scaling is chosen specifically so this holds in practice, not assumed.
inline FixedPointResult solve_fixed_point(const DEQParams &p, const State &x, double tol = 1e-10,
                                           int max_iters = 500) {
  State z(static_cast<size_t>(p.n), 0.0);
  for (int iter = 0; iter < max_iters; ++iter) {
    State z_next = deq_f(p, z, x);
    double diff = 0.0;
    for (int i = 0; i < p.n; ++i) diff = std::max(diff, std::abs(z_next[static_cast<size_t>(i)] - z[static_cast<size_t>(i)]));
    z = z_next;
    if (diff < tol) return {z, iter + 1, true};
  }
  return {z, max_iters, false};
}

inline std::vector<std::vector<double>> jacobian_wrt_z(const DEQParams &p, const State &z, const State &x,
                                                         double eps = 1e-6) {
  std::vector<std::vector<double>> J(static_cast<size_t>(p.n), std::vector<double>(static_cast<size_t>(p.n)));
  for (int j = 0; j < p.n; ++j) {
    State z_plus = z, z_minus = z;
    z_plus[static_cast<size_t>(j)] += eps;
    z_minus[static_cast<size_t>(j)] -= eps;
    State f_plus = deq_f(p, z_plus, x);
    State f_minus = deq_f(p, z_minus, x);
    for (int i = 0; i < p.n; ++i)
      J[static_cast<size_t>(i)][static_cast<size_t>(j)] =
          (f_plus[static_cast<size_t>(i)] - f_minus[static_cast<size_t>(i)]) / (2.0 * eps);
  }
  return J;
}

inline std::vector<std::vector<double>> jacobian_wrt_theta(const DEQParams &p, const State &z, const State &x,
                                                             double eps = 1e-6) {
  std::vector<double> flat = p.flatten();
  int num_params = static_cast<int>(flat.size());
  std::vector<std::vector<double>> J(static_cast<size_t>(p.n), std::vector<double>(static_cast<size_t>(num_params)));
  for (int k = 0; k < num_params; ++k) {
    std::vector<double> flat_plus = flat, flat_minus = flat;
    flat_plus[static_cast<size_t>(k)] += eps;
    flat_minus[static_cast<size_t>(k)] -= eps;
    DEQParams p_plus = DEQParams::from_flat(flat_plus, p.n);
    DEQParams p_minus = DEQParams::from_flat(flat_minus, p.n);
    State f_plus = deq_f(p_plus, z, x);
    State f_minus = deq_f(p_minus, z, x);
    for (int i = 0; i < p.n; ++i)
      J[static_cast<size_t>(i)][static_cast<size_t>(k)] =
          (f_plus[static_cast<size_t>(i)] - f_minus[static_cast<size_t>(i)]) / (2.0 * eps);
  }
  return J;
}

// Implicit-function-theorem backward pass: given dL/dz* (`g`), returns
// dL/dtheta WITHOUT differentiating through the fixed-point iteration.
inline std::vector<double> deq_backward(const DEQParams &p, const State &z_star, const State &x, const State &g) {
  int n = p.n;
  auto Jz = jacobian_wrt_z(p, z_star, x);
  auto Jtheta = jacobian_wrt_theta(p, z_star, x);

  // A = (I - Jz)^T ; solve A v = g.
  std::vector<std::vector<double>> A(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n)));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double identity = (i == j) ? 1.0 : 0.0;
      // (I - Jz)^T [i][j] = (I - Jz)[j][i] = identity_ji - Jz[j][i]
      A[static_cast<size_t>(i)][static_cast<size_t>(j)] = identity - Jz[static_cast<size_t>(j)][static_cast<size_t>(i)];
    }
  std::vector<double> v = detail::solve_linear(A, g);

  int num_params = p.num_params();
  std::vector<double> theta_grad(static_cast<size_t>(num_params), 0.0);
  for (int k = 0; k < num_params; ++k) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += v[static_cast<size_t>(i)] * Jtheta[static_cast<size_t>(i)][static_cast<size_t>(k)];
    theta_grad[static_cast<size_t>(k)] = s;
  }
  return theta_grad;
}

} // namespace sciml
