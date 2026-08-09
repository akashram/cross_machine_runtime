//===- stiffness.h - stability regions + local stiffness ratio ----------===//
//
// PLAN.md Phase 18 step 2: stability and stiffness analysis, built directly
// on step 1's solvers (`sciml::explicit_euler`, `sciml::backward_euler`).
//
// Two pieces:
//   1. A `bounded()` helper that runs a solver and reports whether the
//      trajectory stayed within a sane bound -- the direct way to MEASURE
//      "did this solver blow up" rather than eyeball a printed number.
//      Applied to a stiff linear ODE (dx/dt = lambda*x, lambda very
//      negative) where forward Euler's stability boundary dt < 2/|lambda|
//      is textbook-exact, so the measured pass/fail boundary can be
//      checked against the closed-form prediction directly.
//   2. `local_jacobian_2d()` / `stiffness_ratio_2d()`: a NUMERICALLY
//      estimated (central finite differences, not analytic) local
//      Jacobian and its eigenvalues' real-part spread at a point along a
//      trajectory -- the standard definition of stiffness (Hairer &
//      Wanner: a system is stiff where the Jacobian's eigenvalues have
//      widely separated real parts). Applied to the Van der Pol
//      oscillator, the textbook nonlinear-stiffness example (its
//      stiffness grows with the damping parameter mu).
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../ode_solver/ode_solver.h"

#include <array>
#include <cmath>
#include <complex>

namespace sciml {

// Runs a solver-produced trajectory check: does every state stay within
// `bound` in max-abs-value? Used to measure (not eyeball) whether a solver
// diverged.
inline bool trajectory_stays_bounded(const Trajectory &traj, double bound) {
  for (const State &x : traj.states)
    for (double v : x)
      if (!std::isfinite(v) || std::abs(v) > bound) return false;
  return true;
}

// Central-difference estimate of the 2x2 Jacobian of `f` at (t, x), for a
// 2-dimensional State. Purely numerical -- no analytic derivative of f is
// required from the caller, matching this repo's general preference for
// finite-difference verification over hand-derived Jacobians where the
// system itself is the thing under study (not a component being unit
// tested against a known-correct analytic gradient, as in
// adversarial/input_gradients).
inline std::array<std::array<double, 2>, 2> local_jacobian_2d(const Rhs &f, double t, const State &x,
                                                                double eps = 1e-6) {
  std::array<std::array<double, 2>, 2> J{};
  for (int j = 0; j < 2; ++j) {
    State x_plus = x, x_minus = x;
    x_plus[static_cast<size_t>(j)] += eps;
    x_minus[static_cast<size_t>(j)] -= eps;
    State f_plus = f(t, x_plus);
    State f_minus = f(t, x_minus);
    for (int i = 0; i < 2; ++i)
      J[static_cast<size_t>(i)][static_cast<size_t>(j)] =
          (f_plus[static_cast<size_t>(i)] - f_minus[static_cast<size_t>(i)]) / (2.0 * eps);
  }
  return J;
}

struct EigenPair2D {
  std::complex<double> lambda1, lambda2;
};

// Eigenvalues of a real 2x2 matrix via the closed-form quadratic formula on
// the characteristic polynomial lambda^2 - trace*lambda + det = 0. May be
// complex (a spiraling/oscillatory local mode, expected for an oscillator
// like Van der Pol).
inline EigenPair2D eigenvalues_2d(const std::array<std::array<double, 2>, 2> &J) {
  double trace = J[0][0] + J[1][1];
  double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
  double discriminant = trace * trace - 4.0 * det;
  if (discriminant >= 0.0) {
    double sq = std::sqrt(discriminant);
    return {std::complex<double>((trace + sq) / 2.0, 0.0), std::complex<double>((trace - sq) / 2.0, 0.0)};
  }
  double sq = std::sqrt(-discriminant);
  return {std::complex<double>(trace / 2.0, sq / 2.0), std::complex<double>(trace / 2.0, -sq / 2.0)};
}

// The standard numerical-stiffness measure: ratio of the largest to
// smallest |Re(eigenvalue)| among the local Jacobian's eigenvalues. A
// system is "stiff" at a point where this ratio is large -- one mode
// decays/grows far faster than another, forcing an explicit solver's step
// size down to resolve the fast mode even once it's numerically
// irrelevant to the solution. Guards against a near-zero real part
// (undefined ratio) by returning a sentinel rather than a division blowup.
inline double stiffness_ratio_2d(const EigenPair2D &eig, double floor = 1e-9) {
  double r1 = std::abs(eig.lambda1.real());
  double r2 = std::abs(eig.lambda2.real());
  double lo = std::min(r1, r2);
  double hi = std::max(r1, r2);
  if (lo < floor) return hi / floor; // one mode effectively non-decaying -> report as very stiff
  return hi / lo;
}

} // namespace sciml
