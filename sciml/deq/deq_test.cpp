// deq_test.cpp -- two real checks:
//   1. Fixed-point residual: does z* actually satisfy z* ~= f(z*,x)? The
//      basic correctness check on solve_fixed_point before trusting
//      anything built on top of it.
//   2. Gradient correctness: implicit-function-theorem dL/dtheta compared
//      against finite differences that RE-SOLVE the fixed point from
//      scratch for each perturbed parameter -- same standard as
//      neural_ode's adjoint-gradient check and adversarial/input_gradients.
#include "deq.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace sciml;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

double loss_l2(const State &z, const State &target) {
  double s = 0.0;
  for (size_t i = 0; i < z.size(); ++i) {
    double d = z[i] - target[i];
    s += 0.5 * d * d;
  }
  return s;
}

State loss_grad_l2(const State &z, const State &target) {
  State g(z.size());
  for (size_t i = 0; i < z.size(); ++i) g[i] = z[i] - target[i];
  return g;
}

void test_fixed_point_residual_and_gradients() {
  const int n = 3;
  DEQParams p = DEQParams::random_init(n, /*seed=*/21);
  State x = {0.3, -0.2, 0.5};
  State target = {0.1, 0.1, -0.1};

  FixedPointResult fp = solve_fixed_point(p, x);
  State residual_check = deq_f(p, fp.z_star, x);
  double max_residual = 0.0;
  for (int i = 0; i < n; ++i) max_residual = std::max(max_residual, std::abs(residual_check[static_cast<size_t>(i)] - fp.z_star[static_cast<size_t>(i)]));

  std::printf("  fixed point converged=%s in %d iterations | z*=[%.4f, %.4f, %.4f] | max |f(z*,x)-z*|=%.2e\n",
              fp.converged ? "yes" : "NO", fp.iterations, fp.z_star[0], fp.z_star[1], fp.z_star[2], max_residual);
  require(fp.converged, "fixed-point iteration converges within max_iters");
  require(max_residual < 1e-8, "the converged z* actually satisfies z* = f(z*, x) (residual < 1e-8)");

  State g = loss_grad_l2(fp.z_star, target);
  std::vector<double> analytic_grad = deq_backward(p, fp.z_star, x, g);

  std::vector<double> flat = p.flatten();
  int num_params = static_cast<int>(flat.size());
  const double eps = 1e-4;
  std::vector<double> rel_errs;
  for (int k = 0; k < num_params; ++k) {
    std::vector<double> flat_plus = flat, flat_minus = flat;
    flat_plus[static_cast<size_t>(k)] += eps;
    flat_minus[static_cast<size_t>(k)] -= eps;
    DEQParams p_plus = DEQParams::from_flat(flat_plus, n);
    DEQParams p_minus = DEQParams::from_flat(flat_minus, n);
    FixedPointResult fp_plus = solve_fixed_point(p_plus, x);
    FixedPointResult fp_minus = solve_fixed_point(p_minus, x);
    double loss_plus = loss_l2(fp_plus.z_star, target);
    double loss_minus = loss_l2(fp_minus.z_star, target);
    double numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double rel = std::abs(analytic_grad[static_cast<size_t>(k)] - numeric_grad) / std::max(1e-6, std::abs(numeric_grad));
    rel_errs.push_back(rel);
  }
  std::sort(rel_errs.begin(), rel_errs.end());
  double median = rel_errs[rel_errs.size() / 2];
  double max_err = rel_errs.back();
  std::printf("  gradient check over %d parameters (each re-SOLVES the fixed point from scratch): median relative error=%.6f, max=%.6f\n",
              num_params, median, max_err);
  require(median < 1e-2, "implicit-function-theorem dL/dtheta matches finite differences (re-solving the fixed point each time): median relative error < 1%");
  require(max_err < 5e-2, "every parameter's gradient matches within 5%, not just the median");
}

} // namespace

int main() {
  test_fixed_point_residual_and_gradients();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
