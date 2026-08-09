// neural_ode_test.cpp -- two real checks on the adjoint method, not just
// "it runs":
//   1. Self-consistency: the adjoint backward pass reconstructs x(t) by
//      integrating -f(x) alongside the adjoint variables (no stored
//      forward trajectory). Does the reconstructed x0 actually match the
//      real x0 the forward pass started from? A free correctness check
//      independent of whether the gradient itself is right.
//   2. Gradient correctness: adjoint-computed dL/dtheta compared against
//      finite differences of the ACTUAL loss (rerun the whole forward
//      pass with each parameter perturbed) -- the same standard this
//      repo's other gradient methods (adversarial/input_gradients,
//      backward_euler's Newton Jacobian) are held to.
#include "neural_ode.h"

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

double loss_l2(const State &x_t1, const State &target) {
  double s = 0.0;
  for (size_t i = 0; i < x_t1.size(); ++i) {
    double d = x_t1[i] - target[i];
    s += 0.5 * d * d;
  }
  return s;
}

State loss_grad_l2(const State &x_t1, const State &target) {
  State g(x_t1.size());
  for (size_t i = 0; i < x_t1.size(); ++i) g[i] = x_t1[i] - target[i];
  return g;
}

void test_adjoint_reconstruction_and_gradients() {
  const int n = 1, h = 3;
  const double t0 = 0.0, t1 = 1.0, dt = 0.01;
  NeuralODEParams p = NeuralODEParams::random_init(n, h, /*seed=*/11);
  State x0 = {0.2};
  State target = {0.5};

  State x_t1 = neural_ode_forward(p, x0, t0, t1, dt);
  State dL_dx_t1 = loss_grad_l2(x_t1, target);
  double base_loss = loss_l2(x_t1, target);

  AdjointResult adj = adjoint_backward(p, x_t1, dL_dx_t1, t0, t1, dt);

  double x0_recon_err = std::abs(adj.x0_reconstructed[0] - x0[0]);
  std::printf("  forward x(t1)=%.6f, loss=%.6f | adjoint-reconstructed x0=%.6f (true x0=%.6f), |err|=%.2e\n",
              x_t1[0], base_loss, adj.x0_reconstructed[0], x0[0], x0_recon_err);
  require(x0_recon_err < 1e-3, "adjoint backward pass reconstructs the ORIGINAL x0 (integrating -f(x) backward, no stored forward trajectory) to within 1e-3");

  // Finite-difference gradient check: perturb each flattened parameter,
  // rerun the WHOLE forward pass + loss, compare against the adjoint's
  // theta_grad.
  std::vector<double> flat = p.flatten();
  int num_params = static_cast<int>(flat.size());
  const double eps = 1e-4;
  std::vector<double> rel_errs;
  for (int k = 0; k < num_params; ++k) {
    std::vector<double> flat_plus = flat, flat_minus = flat;
    flat_plus[static_cast<size_t>(k)] += eps;
    flat_minus[static_cast<size_t>(k)] -= eps;
    NeuralODEParams p_plus = NeuralODEParams::from_flat(flat_plus, n, h);
    NeuralODEParams p_minus = NeuralODEParams::from_flat(flat_minus, n, h);
    State x_plus = neural_ode_forward(p_plus, x0, t0, t1, dt);
    State x_minus = neural_ode_forward(p_minus, x0, t0, t1, dt);
    double loss_plus = loss_l2(x_plus, target);
    double loss_minus = loss_l2(x_minus, target);
    double numeric_grad = (loss_plus - loss_minus) / (2.0 * eps);
    double analytic_grad = adj.theta_grad[static_cast<size_t>(k)];
    double rel = std::abs(analytic_grad - numeric_grad) / std::max(1e-6, std::abs(numeric_grad));
    rel_errs.push_back(rel);
  }
  std::sort(rel_errs.begin(), rel_errs.end());
  double median = rel_errs[rel_errs.size() / 2];
  double max_err = rel_errs.back();
  std::printf("  gradient check over %d parameters: median relative error=%.6f, max=%.6f\n", num_params, median,
              max_err);
  require(median < 1e-2, "adjoint-computed dL/dtheta matches finite differences of the actual re-run loss: median relative error < 1%");
  require(max_err < 5e-2, "every parameter's adjoint gradient matches finite differences within 5% (not just the median)");
}

} // namespace

int main() {
  test_adjoint_reconstruction_and_gradients();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
