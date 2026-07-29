// bayesian_opt_test.cpp — real correctness checks: the GP surrogate
// actually interpolates observed points with near-zero uncertainty and
// is more uncertain away from data, Expected Improvement and UCB behave
// per their textbook properties (monotonic in mean and in uncertainty,
// zero EI when no improvement is possible and there's no uncertainty),
// and the full BayesianOptimizer actually converges to a known 1D
// optimum within a modest evaluation budget.
#include "bayesian_opt.h"
#include "gp.h"

#include <cmath>
#include <cstdio>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_gp_interpolates_observed_points() {
  std::vector<std::vector<float>> X = {{0.0f}, {1.0f}, {2.0f}, {3.0f}, {4.0f}};
  std::vector<float> y = {0.0f, 0.841f, 0.909f, 0.141f, -0.757f};  // sin(x) at each point

  GPParams p;
  p.length_scale = 1.0f;
  p.noise_variance = 1e-6f;
  GaussianProcess gp(p);
  gp.fit(X, y);

  float mean, std_dev;
  gp.predict({2.0f}, mean, std_dev);
  std::printf("  at observed x=2.0: predicted mean=%.4f (true=0.909), std=%.6f\n", static_cast<double>(mean), static_cast<double>(std_dev));
  require(std::fabs(mean - 0.909f) < 0.01f, "GP posterior mean matches an observed point's value almost exactly");
  require(std_dev < 0.01f, "GP posterior std is near-zero at an observed point (noise_variance only)");

  float far_mean, far_std;
  gp.predict({10.0f}, far_mean, far_std);
  std::printf("  far from any observed point (x=10.0): std=%.4f\n", static_cast<double>(far_std));
  require(far_std > std_dev, "GP posterior std is larger far from any observed data than at an observed point");
}

void test_expected_improvement_properties() {
  float ei_no_uncertainty = expected_improvement(0.5f, 0.0f, 0.9f, 0.0f);
  float ei_low_mean = expected_improvement(0.5f, 0.1f, 0.9f);
  float ei_high_mean = expected_improvement(0.85f, 0.1f, 0.9f);
  float ei_low_std = expected_improvement(0.5f, 0.05f, 0.9f);
  float ei_high_std = expected_improvement(0.5f, 0.5f, 0.9f);

  std::printf("  EI(mean=0.5,std=0,best=0.9)=%.4f, EI(mean=0.5,std=0.1)=%.4f, EI(mean=0.85,std=0.1)=%.4f\n",
              static_cast<double>(ei_no_uncertainty), static_cast<double>(ei_low_mean), static_cast<double>(ei_high_mean));
  std::printf("  EI(mean=0.5,std=0.05)=%.4f, EI(mean=0.5,std=0.5)=%.4f\n", static_cast<double>(ei_low_std), static_cast<double>(ei_high_std));

  require(ei_no_uncertainty == 0.0f, "EI is exactly zero when there is no uncertainty and the mean can't beat best_so_far");
  require(ei_high_mean > ei_low_mean, "EI increases with a higher predicted mean (holding std fixed)");
  require(ei_high_std > ei_low_std, "EI increases with higher predictive uncertainty (holding mean fixed) -- more uncertainty means more chance of a large improvement");
}

void test_ucb_properties() {
  float ucb_low = upper_confidence_bound(0.5f, 0.1f, 2.0f);
  float ucb_high_mean = upper_confidence_bound(0.7f, 0.1f, 2.0f);
  float ucb_high_std = upper_confidence_bound(0.5f, 0.5f, 2.0f);

  std::printf("  UCB(mean=0.5,std=0.1)=%.4f, UCB(mean=0.7,std=0.1)=%.4f, UCB(mean=0.5,std=0.5)=%.4f\n", static_cast<double>(ucb_low),
              static_cast<double>(ucb_high_mean), static_cast<double>(ucb_high_std));
  require(ucb_high_mean > ucb_low, "UCB increases with a higher predicted mean");
  require(ucb_high_std > ucb_low, "UCB increases with higher predictive uncertainty (kappa rewards exploration)");
}

// Maximize -(x-2)^2 -- a simple, single-optimum function with a known
// answer (x=2, value=0) that a real GP-guided search should get close
// to within a modest budget, not just "run without crashing."
void test_bayesian_optimizer_finds_known_optimum() {
  ObjectiveFn objective = [](const std::vector<float> &x) { return -((x[0] - 2.0f) * (x[0] - 2.0f)); };

  BOParams p;
  p.n_initial_random = 5;
  p.n_iterations = 15;
  p.random_state = 3;
  BayesianOptimizer bo({{-5.0f, 5.0f}}, p);
  auto [best_x, best_value] = bo.optimize(objective);

  std::printf("  found x=%.4f (true optimum x=2.0), value=%.4f (true optimum value=0.0)\n", static_cast<double>(best_x[0]),
              static_cast<double>(best_value));
  require(std::fabs(best_x[0] - 2.0f) < 0.3f, "Bayesian optimization finds a point close to the true optimum within 20 total evaluations");
}

}  // namespace

int main() {
  test_gp_interpolates_observed_points();
  test_expected_improvement_properties();
  test_ucb_properties();
  test_bayesian_optimizer_finds_known_optimum();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
