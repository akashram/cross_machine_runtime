//===- neural_ode.h - NN-parameterized dx/dt, adjoint-method gradients --===//
//
// PLAN.md Phase 18 step 4: Chen, Rubanova, Bettencourt & Duvenaud (2018),
// "Neural Ordinary Differential Equations" -- parameterize dx/dt = f_theta(x)
// with a small network, integrate forward with an existing solver
// (`sciml::rk4`, reused directly, not reimplemented), and compute
// dL/dtheta via the ADJOINT method rather than naive backprop-through-
// every-solver-step: solve one augmented ODE backward in time instead of
// storing every intermediate activation.
//
// Scope note: f_theta(x) is autonomous (no explicit t dependence) -- the
// standard Neural ODE formulation allows dx/dt = f(x,t,theta), but the
// adjoint-method mechanics being verified here (the whole point of this
// step) don't need explicit time-dependence to demonstrate, and dropping
// it keeps the Jacobian bookkeeping below simpler. A real, disclosed scope
// reduction, same pattern as other steps' "simplified vs. the full paper"
// notes.
//
// The adjoint ODE (derivation, reversed time tau = t1 - t so existing
// forward solvers can integrate it -- see neural_ode_test.cpp for the
// finite-difference check this derivation is verified against):
//   dx/dtau     = -f(x)                         (reconstructs x(t) backward, no storage)
//   da/dtau     = +a^T * (df/dx)                 (a(t) = dL/dx(t), the adjoint)
//   dtheta/dtau = +a^T * (df/dtheta)              (accumulates dL/dtheta)
// with initial condition at tau=0 (i.e. t=t1): x=x(t1), a=dL/dx(t1),
// theta_grad=0. At tau=(t1-t0) (i.e. t=t0): theta_grad = dL/dtheta, and x
// should equal the ORIGINAL x(t0) -- a free self-consistency check on the
// reconstruction, independent of whether the gradient itself is right.
//
// REAL BUG, caught by the finite-difference gradient check, not by
// inspection: the hand-derivation above originally concluded
// dtheta/dtau = -a^T*(df/dtheta) (working through the t->tau
// substitution's sign carefully, on paper, looked right). The actual
// gradient check against re-run finite differences came back with EVERY
// parameter's relative error at exactly 2.0 -- the unmistakable signature
// of `numeric = -analytic` -- not a subtle Jacobian bug. Flipping that one
// sign (to the `+` shown above) brought every parameter's relative error
// to ~0. Left as a "trust the test over the hand derivation" case study:
// the sign convention for dL/dtheta's line integral is genuinely easy to
// flip once during a t->tau substitution, and the finite-difference check
// is what actually catches it, not re-reading the algebra.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../ode_solver/ode_solver.h"

#include <cmath>
#include <random>

namespace sciml {

// f_theta(x) = W2 * tanh(W1*x + b1) + b2. x in R^n, hidden width h.
struct NeuralODEParams {
  int n = 1, h = 4;
  std::vector<double> W1, b1, W2, b2; // sizes h*n, h, n*h, n

  int num_params() const { return h * n + h + n * h + n; }

  static NeuralODEParams random_init(int n, int h, uint64_t seed) {
    NeuralODEParams p;
    p.n = n;
    p.h = h;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> d(0.0, 0.5);
    p.W1.resize(static_cast<size_t>(h * n));
    p.b1.resize(static_cast<size_t>(h));
    p.W2.resize(static_cast<size_t>(n * h));
    p.b2.resize(static_cast<size_t>(n));
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

  static NeuralODEParams from_flat(const std::vector<double> &flat, int n, int h) {
    NeuralODEParams p;
    p.n = n;
    p.h = h;
    size_t off = 0;
    p.W1.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(h * n)));
    off += static_cast<size_t>(h * n);
    p.b1.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(h)));
    off += static_cast<size_t>(h);
    p.W2.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n * h)));
    off += static_cast<size_t>(n * h);
    p.b2.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n)));
    return p;
  }
};

// f_theta(x) = W2 * tanh(W1*x + b1) + b2.
inline State neural_ode_f(const NeuralODEParams &p, const State &x) {
  std::vector<double> pre(static_cast<size_t>(p.h), 0.0), hid(static_cast<size_t>(p.h), 0.0);
  for (int i = 0; i < p.h; ++i) {
    double s = p.b1[static_cast<size_t>(i)];
    for (int j = 0; j < p.n; ++j) s += p.W1[static_cast<size_t>(i * p.n + j)] * x[static_cast<size_t>(j)];
    pre[static_cast<size_t>(i)] = s;
    hid[static_cast<size_t>(i)] = std::tanh(s);
  }
  State out(static_cast<size_t>(p.n), 0.0);
  for (int i = 0; i < p.n; ++i) {
    double s = p.b2[static_cast<size_t>(i)];
    for (int j = 0; j < p.h; ++j) s += p.W2[static_cast<size_t>(i * p.h + j)] * hid[static_cast<size_t>(j)];
    out[static_cast<size_t>(i)] = s;
  }
  return out;
}

// df/dx, an n x n numerical Jacobian via central differences.
inline std::vector<std::vector<double>> jacobian_wrt_x(const NeuralODEParams &p, const State &x, double eps = 1e-6) {
  std::vector<std::vector<double>> J(static_cast<size_t>(p.n), std::vector<double>(static_cast<size_t>(p.n)));
  for (int j = 0; j < p.n; ++j) {
    State x_plus = x, x_minus = x;
    x_plus[static_cast<size_t>(j)] += eps;
    x_minus[static_cast<size_t>(j)] -= eps;
    State f_plus = neural_ode_f(p, x_plus);
    State f_minus = neural_ode_f(p, x_minus);
    for (int i = 0; i < p.n; ++i)
      J[static_cast<size_t>(i)][static_cast<size_t>(j)] =
          (f_plus[static_cast<size_t>(i)] - f_minus[static_cast<size_t>(i)]) / (2.0 * eps);
  }
  return J;
}

// df/dtheta, an n x num_params numerical Jacobian via central differences
// over the FLATTENED parameter vector.
inline std::vector<std::vector<double>> jacobian_wrt_theta(const NeuralODEParams &p, const State &x,
                                                             double eps = 1e-6) {
  std::vector<double> flat = p.flatten();
  int num_params = static_cast<int>(flat.size());
  std::vector<std::vector<double>> J(static_cast<size_t>(p.n), std::vector<double>(static_cast<size_t>(num_params)));
  for (int k = 0; k < num_params; ++k) {
    std::vector<double> flat_plus = flat, flat_minus = flat;
    flat_plus[static_cast<size_t>(k)] += eps;
    flat_minus[static_cast<size_t>(k)] -= eps;
    NeuralODEParams p_plus = NeuralODEParams::from_flat(flat_plus, p.n, p.h);
    NeuralODEParams p_minus = NeuralODEParams::from_flat(flat_minus, p.n, p.h);
    State f_plus = neural_ode_f(p_plus, x);
    State f_minus = neural_ode_f(p_minus, x);
    for (int i = 0; i < p.n; ++i)
      J[static_cast<size_t>(i)][static_cast<size_t>(k)] =
          (f_plus[static_cast<size_t>(i)] - f_minus[static_cast<size_t>(i)]) / (2.0 * eps);
  }
  return J;
}

// Forward pass: integrate dx/dt = f_theta(x) from t0 to t1 via the
// existing sciml::rk4 solver (direct reuse, not a copy).
inline State neural_ode_forward(const NeuralODEParams &p, State x0, double t0, double t1, double dt) {
  Rhs f = [&p](double, const State &x) { return neural_ode_f(p, x); };
  Trajectory traj = rk4(f, std::move(x0), t0, t1, dt);
  return traj.back();
}

struct AdjointResult {
  State x0_reconstructed;
  std::vector<double> theta_grad;
};

// Adjoint backward pass -- see the file header for the derivation. Reuses
// sciml::rk4 by integrating the augmented [x, a, theta_grad] system
// forward in reversed time tau = t1 - t.
inline AdjointResult adjoint_backward(const NeuralODEParams &p, const State &x_t1, const State &dL_dx_t1, double t0,
                                       double t1, double dt) {
  int n = p.n;
  int num_params = p.num_params();

  State y0(static_cast<size_t>(2 * n + num_params), 0.0);
  for (int i = 0; i < n; ++i) y0[static_cast<size_t>(i)] = x_t1[static_cast<size_t>(i)];
  for (int i = 0; i < n; ++i) y0[static_cast<size_t>(n + i)] = dL_dx_t1[static_cast<size_t>(i)];
  // theta_grad portion starts at zero (default-initialized above).

  Rhs augmented = [&p, n, num_params](double, const State &y) -> State {
    State x(y.begin(), y.begin() + n);
    State a(y.begin() + n, y.begin() + 2 * n);

    State fx = neural_ode_f(p, x);
    auto Jx = jacobian_wrt_x(p, x);
    auto Jtheta = jacobian_wrt_theta(p, x);

    State dy(static_cast<size_t>(2 * n + num_params), 0.0);
    for (int i = 0; i < n; ++i) dy[static_cast<size_t>(i)] = -fx[static_cast<size_t>(i)];
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int i = 0; i < n; ++i) s += a[static_cast<size_t>(i)] * Jx[static_cast<size_t>(i)][static_cast<size_t>(j)];
      dy[static_cast<size_t>(n + j)] = s;
    }
    for (int k = 0; k < num_params; ++k) {
      double s = 0.0;
      for (int i = 0; i < n; ++i) s += a[static_cast<size_t>(i)] * Jtheta[static_cast<size_t>(i)][static_cast<size_t>(k)];
      dy[static_cast<size_t>(2 * n + k)] = s;
    }
    return dy;
  };

  Trajectory traj = rk4(augmented, y0, 0.0, t1 - t0, dt);
  const State &y_final = traj.back();

  AdjointResult result;
  result.x0_reconstructed.assign(y_final.begin(), y_final.begin() + n);
  result.theta_grad.assign(y_final.begin() + 2 * n, y_final.end());
  return result;
}

} // namespace sciml
