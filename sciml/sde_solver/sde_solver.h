//===- sde_solver.h - Euler-Maruyama and Milstein SDE solvers ------------===//
//
// PLAN.md Phase 18 step 3: two fixed-step solvers for scalar stochastic
// differential equations dx = a(t,x) dt + b(t,x) dW, where W is a standard
// Wiener process:
//
//   - Euler-Maruyama: x_{n+1} = x_n + a(t_n,x_n)*dt + b(t_n,x_n)*dW_n.
//     Strong order 0.5, weak order 1.0 (Kloeden & Platen 1992).
//   - Milstein: adds the correction term
//     0.5 * b(t_n,x_n) * b'(t_n,x_n) * (dW_n^2 - dt), where b' is the
//     diffusion coefficient's derivative w.r.t. x, estimated here via
//     central finite differences (no analytic derivative required from
//     the caller, same convention as `stiffness.h`'s numerical Jacobian
//     and `ode_solver.h::backward_euler`'s numerical Jacobian). Strong
//     order 1.0 -- a real accuracy improvement over Euler-Maruyama, but
//     ONLY when b actually depends on x (multiplicative/state-dependent
//     noise); for additive noise (b constant in x, e.g. an
//     Ornstein-Uhlenbeck process), b'=0 and Milstein reduces exactly to
//     Euler-Maruyama.
//
// Both solvers take a PRE-GENERATED sequence of standard normal increments
// (`generate_normal_increments`) rather than an RNG, so that Euler-Maruyama
// and Milstein can be run on the exact same underlying Brownian path -- the
// only way to measure STRONG (pathwise) error meaningfully; comparing
// against two independently-sampled paths would only measure WEAK
// (distributional) error, which doesn't distinguish the two methods'
// actual strong-order-of-convergence difference.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cmath>
#include <functional>
#include <random>
#include <vector>

namespace sciml {

using DriftFn = std::function<double(double t, double x)>;
using DiffusionFn = std::function<double(double t, double x)>;

struct SDETrajectory {
  std::vector<double> times;
  std::vector<double> states;
  std::vector<double> brownian; // cumulative W(t) at each time point, W(t0)=0
};

// One standard-normal increment per step. Fixing this sequence upfront
// (rather than drawing lazily inside each solver) is what makes
// Euler-Maruyama and Milstein directly comparable path-by-path.
inline std::vector<double> generate_normal_increments(int num_steps, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::vector<double> out(static_cast<size_t>(num_steps));
  for (double &v : out) v = z(rng);
  return out;
}

// `t1` is accepted (rather than derived from `z.size()*dt`) purely for
// signature symmetry with ode_solver.h's `explicit_euler`/`rk4`/
// `backward_euler` -- it isn't otherwise used, since `z`'s length already
// fixes the number of steps.
inline SDETrajectory euler_maruyama(const DriftFn &a, const DiffusionFn &b, double x0, double t0, double t1,
                                     double dt, const std::vector<double> &z) {
  (void)t1;
  SDETrajectory traj;
  double t = t0, x = x0, W = 0.0;
  traj.times.push_back(t);
  traj.states.push_back(x);
  traj.brownian.push_back(W);
  double sqrt_dt = std::sqrt(dt);
  for (double zi : z) {
    double dW = sqrt_dt * zi;
    x = x + a(t, x) * dt + b(t, x) * dW;
    W += dW;
    t += dt;
    traj.times.push_back(t);
    traj.states.push_back(x);
    traj.brownian.push_back(W);
  }
  return traj;
}

inline SDETrajectory milstein(const DriftFn &a, const DiffusionFn &b, double x0, double t0, double t1, double dt,
                               const std::vector<double> &z, double eps = 1e-6) {
  (void)t1;
  SDETrajectory traj;
  double t = t0, x = x0, W = 0.0;
  traj.times.push_back(t);
  traj.states.push_back(x);
  traj.brownian.push_back(W);
  double sqrt_dt = std::sqrt(dt);
  for (double zi : z) {
    double dW = sqrt_dt * zi;
    double bx = b(t, x);
    double b_prime = (b(t, x + eps) - b(t, x - eps)) / (2.0 * eps);
    x = x + a(t, x) * dt + bx * dW + 0.5 * bx * b_prime * (dW * dW - dt);
    W += dW;
    t += dt;
    traj.times.push_back(t);
    traj.states.push_back(x);
    traj.brownian.push_back(W);
  }
  return traj;
}

} // namespace sciml
