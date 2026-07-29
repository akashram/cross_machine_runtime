// tpe_test.cpp — real correctness checks: the Parzen-window density
// estimate is actually higher near a cluster of observed points than
// far away, sampling from it actually concentrates near that cluster,
// the single-point bandwidth heuristic matches Bergstra et al.'s
// definition exactly, and the full TPEOptimizer converges to a known
// 1D optimum within a modest budget (same function bayesian_opt_test.cpp
// uses, for a fair side-by-side comparison of both optimizers'
// correctness).
#include "tpe.h"

#include <cmath>
#include <cstdio>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_parzen_density_higher_near_cluster() {
  std::vector<float> values = {-0.1f, 0.0f, 0.1f};
  Parzen1D p = build_parzen1d(values, -10.0f, 10.0f);

  float density_near = parzen1d_density(p, 0.0f);
  float density_far = parzen1d_density(p, 8.0f);

  std::printf("  density near cluster (x=0)=%.6f, density far away (x=8)=%.6f\n", static_cast<double>(density_near),
              static_cast<double>(density_far));
  require(density_near > density_far, "Parzen density is higher near a cluster of observed points than far away");
}

void test_parzen_sample_concentrates_near_points() {
  std::vector<float> values = {2.9f, 3.0f, 3.1f};
  Parzen1D p = build_parzen1d(values, -10.0f, 10.0f);
  std::mt19937 rng(1);

  double sum = 0.0;
  int n_samples = 2000;
  for (int i = 0; i < n_samples; ++i) sum += static_cast<double>(parzen1d_sample(p, rng, -10.0f, 10.0f));
  double mean = sum / n_samples;

  std::printf("  mean of %d samples from a Parzen built near x=3.0: %.4f\n", n_samples, mean);
  require(std::fabs(mean - 3.0) < 0.3, "sampling from the Parzen density concentrates near the cluster it was built from");
}

void test_single_point_bandwidth_is_full_range() {
  Parzen1D p = build_parzen1d({5.0f}, 0.0f, 10.0f);
  std::printf("  single-point bandwidth=%.4f (range=10.0)\n", static_cast<double>(p.bandwidths[0]));
  require(std::fabs(p.bandwidths[0] - 10.0f) < 1e-4f, "a single-point Parzen group uses the full bound range as its bandwidth (Bergstra et al.'s heuristic)");
}

void test_tpe_finds_known_optimum() {
  ObjectiveFn objective = [](const std::vector<float> &x) { return -((x[0] - 2.0f) * (x[0] - 2.0f)); };

  TPEParams p;
  p.n_initial_random = 5;
  p.n_iterations = 15;
  p.random_state = 3;
  TPEOptimizer tpe({{-5.0f, 5.0f}}, p);
  auto [best_x, best_value] = tpe.optimize(objective);

  std::printf("  found x=%.4f (true optimum x=2.0), value=%.4f (true optimum value=0.0)\n", static_cast<double>(best_x[0]),
              static_cast<double>(best_value));
  require(std::fabs(best_x[0] - 2.0f) < 0.3f, "TPE finds a point close to the true optimum within 20 total evaluations");
}

}  // namespace

int main() {
  test_parzen_density_higher_near_cluster();
  test_parzen_sample_concentrates_near_points();
  test_single_point_bandwidth_is_full_range();
  test_tpe_finds_known_optimum();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
