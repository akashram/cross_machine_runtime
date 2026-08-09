//===- ode_solver.h - explicit Euler, RK4, implicit backward Euler ------===//
//
// PLAN.md Phase 18 step 1: ODE solver library. Three fixed-step solvers for
// initial value problems dx/dt = f(t, x), x(t0) = x0:
//
//   - Explicit (forward) Euler: x_{n+1} = x_n + dt * f(t_n, x_n).
//     O(dt) global error (first order); the cheapest possible step, but
//     unstable for stiff systems at large dt -- see step 2.
//   - Classical RK4: the standard four-stage Runge-Kutta method.
//     O(dt^4) global error for a smooth f; the default "just works" choice
//     for non-stiff systems, at 4x the per-step cost of Euler.
//   - Implicit (backward) Euler: x_{n+1} = x_n + dt * f(t_{n+1}, x_{n+1}).
//     O(dt) global error like forward Euler (same order, NOT more
//     accurate), but A-stable: it doesn't blow up at large dt on stiff
//     systems the way forward Euler does (measured directly in step 2).
//     Each step solves a (generally nonlinear) system for x_{n+1} via
//     Newton's method with a finite-difference Jacobian -- no analytic
//     Jacobian required from the caller, at the cost of extra f()
//     evaluations per Newton iteration.
//
// State type is std::vector<double> throughout so the same three solvers
// work unmodified for scalar systems (exponential decay), 2D systems
// (harmonic oscillator, Van der Pol in step 2), and eventually the
// network-parameterized dx/dt of step 4's Neural ODEs.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace sciml {

using State = std::vector<double>;
// Right-hand side of dx/dt = f(t, x).
using Rhs = std::function<State(double t, const State &x)>;

struct Trajectory {
  std::vector<double> times;
  std::vector<State> states;

  const State &back() const { return states.back(); }
};

inline State axpy(double a, const State &x, const State &y) {
  State out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = a * x[i] + y[i];
  return out;
}

inline State add(const State &x, const State &y) {
  State out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] + y[i];
  return out;
}

inline State scale(double a, const State &x) {
  State out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = a * x[i];
  return out;
}

// Explicit forward Euler. O(dt) global error.
inline Trajectory explicit_euler(const Rhs &f, State x0, double t0, double t1, double dt) {
  Trajectory traj;
  double t = t0;
  State x = std::move(x0);
  traj.times.push_back(t);
  traj.states.push_back(x);
  while (t < t1 - 1e-12) {
    double step = std::min(dt, t1 - t);
    State k1 = f(t, x);
    x = axpy(step, k1, x);
    t += step;
    traj.times.push_back(t);
    traj.states.push_back(x);
  }
  return traj;
}

// Classical 4th-order Runge-Kutta. O(dt^4) global error for smooth f.
inline Trajectory rk4(const Rhs &f, State x0, double t0, double t1, double dt) {
  Trajectory traj;
  double t = t0;
  State x = std::move(x0);
  traj.times.push_back(t);
  traj.states.push_back(x);
  while (t < t1 - 1e-12) {
    double h = std::min(dt, t1 - t);
    State k1 = f(t, x);
    State k2 = f(t + h / 2.0, axpy(h / 2.0, k1, x));
    State k3 = f(t + h / 2.0, axpy(h / 2.0, k2, x));
    State k4 = f(t + h, axpy(h, k3, x));
    State sum(x.size());
    for (size_t i = 0; i < x.size(); ++i) sum[i] = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    x = axpy(h, sum, x);
    t += h;
    traj.times.push_back(t);
    traj.states.push_back(x);
  }
  return traj;
}

namespace detail {

// Solve A y = b for y via Gaussian elimination with partial pivoting.
// Small, dense, unoptimized -- sized for the Newton solves below (state
// dimensions in the single digits throughout this repo's usage).
inline std::vector<double> solve_linear(std::vector<std::vector<double>> A, std::vector<double> b) {
  size_t n = b.size();
  for (size_t col = 0; col < n; ++col) {
    size_t pivot = col;
    double best = std::abs(A[col][col]);
    for (size_t row = col + 1; row < n; ++row) {
      if (std::abs(A[row][col]) > best) {
        best = std::abs(A[row][col]);
        pivot = row;
      }
    }
    if (best < 1e-300) throw std::runtime_error("solve_linear: singular matrix");
    std::swap(A[col], A[pivot]);
    std::swap(b[col], b[pivot]);
    for (size_t row = col + 1; row < n; ++row) {
      double factor = A[row][col] / A[col][col];
      for (size_t k = col; k < n; ++k) A[row][k] -= factor * A[col][k];
      b[row] -= factor * b[col];
    }
  }
  std::vector<double> y(n);
  for (size_t i = n; i-- > 0;) {
    double sum = b[i];
    for (size_t k = i + 1; k < n; ++k) sum -= A[i][k] * y[k];
    y[i] = sum / A[i][i];
  }
  return y;
}

} // namespace detail

// Implicit backward Euler. Each step solves
//   g(y) = y - x_n - dt * f(t_{n+1}, y) = 0
// for y = x_{n+1} via Newton's method with a central-difference Jacobian
// of g (equivalently, I - dt * J_f). O(dt) global error like forward
// Euler -- backward Euler's value is stability at large dt on stiff
// systems (step 2), not higher accuracy.
inline Trajectory backward_euler(const Rhs &f, State x0, double t0, double t1, double dt,
                                  double newton_tol = 1e-10, int max_newton_iter = 50) {
  Trajectory traj;
  double t = t0;
  State x = std::move(x0);
  traj.times.push_back(t);
  traj.states.push_back(x);
  const double eps = 1e-6;
  while (t < t1 - 1e-12) {
    double h = std::min(dt, t1 - t);
    double t_next = t + h;
    State x_n = x;      // known state at start of step
    State y = x_n;       // Newton iterate for x_{n+1}, initialized at x_n

    auto g = [&](const State &y_) {
      State fy = f(t_next, y_);
      return axpy(-h, fy, State(x_n.size() ? add(y_, scale(-1.0, x_n)) : y_));
    };

    for (int iter = 0; iter < max_newton_iter; ++iter) {
      State gy = g(y);
      double norm = 0.0;
      for (double v : gy) norm += v * v;
      norm = std::sqrt(norm);
      if (norm < newton_tol) break;

      size_t n = y.size();
      std::vector<std::vector<double>> J(n, std::vector<double>(n));
      for (size_t j = 0; j < n; ++j) {
        State y_plus = y, y_minus = y;
        y_plus[j] += eps;
        y_minus[j] -= eps;
        State g_plus = g(y_plus);
        State g_minus = g(y_minus);
        for (size_t i = 0; i < n; ++i) J[i][j] = (g_plus[i] - g_minus[i]) / (2.0 * eps);
      }
      std::vector<double> neg_gy(n);
      for (size_t i = 0; i < n; ++i) neg_gy[i] = -gy[i];
      std::vector<double> delta = detail::solve_linear(J, neg_gy);
      for (size_t i = 0; i < n; ++i) y[i] += delta[i];
    }
    x = y;
    t = t_next;
    traj.times.push_back(t);
    traj.states.push_back(x);
  }
  return traj;
}

} // namespace sciml
