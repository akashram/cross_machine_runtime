// ode_solver_test.cpp -- verifies all three solvers against ODEs with known
// closed-form solutions (not just "it runs"):
//   1. Exponential decay (dx/dt = -k x): linear, tests basic correctness of
//      all three solvers directly against x(t) = x0 * exp(-k*t).
//   2. Harmonic oscillator (2D linear system): tests RK4's empirical 4th-
//      order convergence rate directly (halve dt, error should drop ~16x),
//      vs. forward/backward Euler's ~2x (1st order).
//   3. Logistic growth (dx/dt = r*x*(1-x/K)): genuinely NONLINEAR RHS, so
//      backward Euler's implicit equation is nonlinear in y -- this is the
//      test that actually exercises Newton's method inside backward_euler,
//      not just its linear-equation special case.
#include "ode_solver.h"

#include <cmath>
#include <cstdio>

using namespace sciml;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

double rel_err(double got, double want) { return std::abs(got - want) / std::max(1e-12, std::abs(want)); }

void test_exponential_decay() {
  const double k = 1.3, x0 = 2.0, t1 = 2.0, dt = 0.001;
  Rhs f = [k](double, const State &x) -> State { return {-k * x[0]}; };
  double closed_form = x0 * std::exp(-k * t1);

  Trajectory te = explicit_euler(f, {x0}, 0.0, t1, dt);
  Trajectory tr = rk4(f, {x0}, 0.0, t1, dt);
  Trajectory tb = backward_euler(f, {x0}, 0.0, t1, dt);

  double err_euler = rel_err(te.back()[0], closed_form);
  double err_rk4 = rel_err(tr.back()[0], closed_form);
  double err_backward = rel_err(tb.back()[0], closed_form);
  std::printf("  closed form x(2.0) = %.6f | euler err=%.2e rk4 err=%.2e backward err=%.2e\n", closed_form,
              err_euler, err_rk4, err_backward);

  // 1st-order global error at dt=1e-3 over a length-2.0 interval on this
  // ODE (k=1.3) is genuinely ~1.7e-3, not <1e-3 -- O(dt) has a k-dependent
  // prefactor, and the first version of this test set an unrealistically
  // tight bound that both 1st-order methods (correctly) failed.
  require(err_euler < 5e-3, "explicit_euler matches closed-form exponential decay");
  require(err_rk4 < 1e-9, "rk4 matches closed-form exponential decay to near machine precision at dt=1e-3");
  require(err_backward < 5e-3, "backward_euler matches closed-form exponential decay");
  // RK4 must be dramatically more accurate than either 1st-order method at
  // the same step size -- the entire point of higher order.
  require(err_rk4 < err_euler * 1e-4, "rk4 is orders of magnitude more accurate than explicit_euler at identical dt");
}

void test_rk4_fourth_order_convergence() {
  // Harmonic oscillator: x'' = -omega^2 x, as the first-order system
  // x0' = x1, x1' = -omega^2 x0. Closed form: x0(t) = A cos(wt) + B sin(wt).
  const double omega = 2.0;
  Rhs f = [omega](double, const State &x) -> State { return {x[1], -omega * omega * x[0]}; };
  State x0 = {1.0, 0.0}; // A=1, B=0 -> x0(t) = cos(omega t)
  double t1 = 1.0;
  auto closed_form = [&](double t) { return std::cos(omega * t); };

  auto measure_error = [&](double dt) {
    Trajectory tr = rk4(f, x0, 0.0, t1, dt);
    return std::abs(tr.back()[0] - closed_form(t1));
  };
  auto measure_error_euler = [&](double dt) {
    Trajectory te = explicit_euler(f, x0, 0.0, t1, dt);
    return std::abs(te.back()[0] - closed_form(t1));
  };

  double dt1 = 0.02, dt2 = 0.01;
  double e_rk4_1 = measure_error(dt1), e_rk4_2 = measure_error(dt2);
  double e_euler_1 = measure_error_euler(dt1), e_euler_2 = measure_error_euler(dt2);
  double ratio_rk4 = e_rk4_1 / std::max(1e-16, e_rk4_2);
  double ratio_euler = e_euler_1 / std::max(1e-16, e_euler_2);
  std::printf("  halving dt: rk4 error ratio=%.2f (expect ~16, 4th order) | euler error ratio=%.2f (expect ~2, 1st order)\n",
              ratio_rk4, ratio_euler);

  // Empirical order p satisfies error ratio ~ 2^p when dt halves. RK4 (p=4)
  // should show a ratio well above Euler's (p=1), even allowing generous
  // slack for a coarse, non-asymptotic dt.
  require(ratio_rk4 > 10.0, "rk4 shows ~4th-order convergence (error ratio > 10 when dt halves)");
  require(ratio_euler > 1.5 && ratio_euler < 4.0, "explicit_euler shows ~1st-order convergence (error ratio ~2 when dt halves)");
}

void test_logistic_growth_nonlinear_backward_euler() {
  // dx/dt = r*x*(1-x/K); closed form x(t) = K / (1 + ((K-x0)/x0)*exp(-r*t)).
  const double r = 1.5, K = 10.0, x0 = 1.0, t1 = 3.0, dt = 0.01;
  Rhs f = [r, K](double, const State &x) -> State { return {r * x[0] * (1.0 - x[0] / K)}; };
  auto closed_form = [&](double t) { return K / (1.0 + ((K - x0) / x0) * std::exp(-r * t)); };

  Trajectory tb = backward_euler(f, {x0}, 0.0, t1, dt);
  Trajectory tr = rk4(f, {x0}, 0.0, t1, dt);
  double want = closed_form(t1);
  double err_backward = rel_err(tb.back()[0], want);
  double err_rk4 = rel_err(tr.back()[0], want);
  std::printf("  closed form logistic x(3.0) = %.6f | backward_euler err=%.2e rk4 err=%.2e\n", want, err_backward,
              err_rk4);

  require(err_backward < 1e-2, "backward_euler's Newton solve converges on a genuinely NONLINEAR implicit equation and matches closed-form logistic growth");
  require(err_rk4 < 1e-6, "rk4 matches closed-form logistic growth");
}

} // namespace

int main() {
  test_exponential_decay();
  test_rk4_fourth_order_convergence();
  test_logistic_growth_nonlinear_backward_euler();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
