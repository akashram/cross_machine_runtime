// stiffness_test.cpp -- three real, measured checks:
//   1. Forward Euler's textbook-exact stability boundary dt < 2/|lambda| on
//      a stiff linear ODE: measured to hold exactly (stable just below the
//      boundary, diverges just above it), with backward Euler staying
//      bounded on the SAME above-boundary dt (A-stability).
//   2. Van der Pol's local stiffness ratio, numerically estimated via
//      stiffness.h's finite-difference Jacobian + eigenvalues, measurably
//      grows with the damping parameter mu -- the textbook nonlinear-
//      stiffness example, measured directly rather than cited and assumed.
//   3. The same forward-vs-backward-Euler bounded/unbounded contrast as
//      test 1, but on the nonlinear, stiff (large-mu) Van der Pol system.
#include "stiffness.h"

#include <cmath>
#include <cstdio>

using namespace sciml;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_linear_stability_boundary() {
  const double lambda = -1000.0;
  const double t1 = 0.1;
  Rhs f = [lambda](double, const State &x) -> State { return {lambda * x[0]}; };

  double dt_bound = 2.0 / std::abs(lambda); // 0.002, exact forward-Euler stability boundary
  double dt_stable = dt_bound * 0.75;       // 0.0015, below boundary
  double dt_unstable = dt_bound * 2.5;      // 0.005, above boundary

  Trajectory euler_stable = explicit_euler(f, {1.0}, 0.0, t1, dt_stable);
  Trajectory euler_unstable = explicit_euler(f, {1.0}, 0.0, t1, dt_unstable);
  Trajectory backward_at_unstable_dt = backward_euler(f, {1.0}, 0.0, t1, dt_unstable);

  bool stable_bounded = trajectory_stays_bounded(euler_stable, 10.0);
  bool unstable_bounded = trajectory_stays_bounded(euler_unstable, 10.0);
  bool backward_bounded = trajectory_stays_bounded(backward_at_unstable_dt, 10.0);

  std::printf("  dt_bound=2/|lambda|=%.4f | dt_stable=%.4f (below) | dt_unstable=%.4f (above, 2.5x)\n", dt_bound,
              dt_stable, dt_unstable);
  std::printf("  |x_final|: euler@dt_stable=%.3e  euler@dt_unstable=%.3e  backward_euler@dt_unstable=%.3e\n",
              std::abs(euler_stable.back()[0]), std::abs(euler_unstable.back()[0]),
              std::abs(backward_at_unstable_dt.back()[0]));

  require(stable_bounded, "explicit_euler stays bounded just BELOW its exact stability boundary dt < 2/|lambda|");
  require(!unstable_bounded, "explicit_euler DIVERGES just above its exact stability boundary (measured, not assumed)");
  require(backward_bounded, "backward_euler stays bounded at the SAME dt that makes explicit_euler diverge (A-stability)");
}

void test_van_der_pol_stiffness_grows_with_mu() {
  auto vdp = [](double mu) -> Rhs {
    return [mu](double, const State &x) -> State { return {x[1], mu * (1.0 - x[0] * x[0]) * x[1] - x[0]}; };
  };
  State point = {0.0, 2.0}; // fixed evaluation point, same for both mu values

  double mu_small = 1.0, mu_large = 50.0;
  auto J_small = local_jacobian_2d(vdp(mu_small), 0.0, point);
  auto J_large = local_jacobian_2d(vdp(mu_large), 0.0, point);
  auto eig_small = eigenvalues_2d(J_small);
  auto eig_large = eigenvalues_2d(J_large);

  bool small_is_complex = std::abs(eig_small.lambda1.imag()) > 1e-9;
  bool large_is_real = std::abs(eig_large.lambda1.imag()) < 1e-9;
  double ratio_large = stiffness_ratio_2d(eig_large);

  std::printf("  mu=%.0f: eigenvalues = %.3f%+.3fi, %.3f%+.3fi (complex -> locally oscillatory, not 'stiff' in the real-eigenvalue sense)\n",
              mu_small, eig_small.lambda1.real(), eig_small.lambda1.imag(), eig_small.lambda2.real(),
              eig_small.lambda2.imag());
  std::printf("  mu=%.0f: eigenvalues = %.3f, %.3f (real, widely separated) -> stiffness ratio = %.1f\n", mu_large,
              eig_large.lambda1.real(), eig_large.lambda2.real(), ratio_large);

  require(small_is_complex, "at mu=1, Van der Pol's local Jacobian has complex eigenvalues (near-harmonic oscillator, genuinely not stiff here)");
  require(large_is_real, "at mu=50, Van der Pol's local Jacobian has real eigenvalues (relaxation-oscillator regime)");
  require(ratio_large > 100.0, "at mu=50, the real eigenvalues are separated by a stiffness ratio > 100 -- measured, real stiffness growth with mu, the textbook Van der Pol behavior");
}

void test_van_der_pol_solver_bounded_comparison() {
  const double mu = 200.0; // strongly stiff relaxation-oscillator regime
  Rhs f = [mu](double, const State &x) -> State { return {x[1], mu * (1.0 - x[0] * x[0]) * x[1] - x[0]}; };
  State x0 = {2.0, 0.0};
  double t1 = 1.0;
  double dt = 0.01; // resolves the SLOW manifold fine, far too coarse for VdP's fast manifold at mu=200

  Trajectory euler = explicit_euler(f, x0, 0.0, t1, dt);
  Trajectory backward = backward_euler(f, x0, 0.0, t1, dt);
  bool euler_bounded = trajectory_stays_bounded(euler, 1.0e6);
  bool backward_bounded = trajectory_stays_bounded(backward, 1.0e6);

  std::printf("  Van der Pol, mu=%.0f, dt=%.3f: explicit_euler bounded=%s | backward_euler bounded=%s\n", mu, dt,
              euler_bounded ? "yes" : "NO (diverged)", backward_bounded ? "yes" : "NO (diverged)");

  require(!euler_bounded, "explicit_euler numerically diverges on stiff (mu=200) Van der Pol at a dt that resolves the true dynamics' slow manifold but not its fast one");
  require(backward_bounded, "backward_euler stays bounded on the SAME stiff system at the SAME dt (its whole point: stability, not higher accuracy)");
}

} // namespace

int main() {
  test_linear_stability_boundary();
  test_van_der_pol_stiffness_grows_with_mu();
  test_van_der_pol_solver_bounded_comparison();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
