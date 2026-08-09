// crossbar_mac_test.cpp -- measures crossbar MAC accuracy against the exact
// digital reference, not just "the code runs":
//   1. Basic correctness at moderate precision/noise.
//   2. Precision sweep (num_levels) at FIXED crossbar size and FIXED
//      random W/x -- isolates the effect of quantization/write-noise
//      granularity alone.
//   3. Crossbar-size sweep at fixed precision -- reports the real,
//      measured relationship between crossbar size and relative MAC
//      error (not assumed to improve OR degrade with scale; whichever
//      direction the numbers actually show).
#include "crossbar_mac.h"

#include <cstdio>
#include <random>

using namespace analog;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

Matrix random_matrix(int rows, int cols, std::mt19937 &rng) {
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  Matrix W(static_cast<size_t>(rows), std::vector<double>(static_cast<size_t>(cols)));
  for (auto &row : W)
    for (double &v : row) v = d(rng);
  return W;
}

std::vector<double> random_vec(int n, std::mt19937 &rng) {
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<double> x(static_cast<size_t>(n));
  for (double &v : x) v = d(rng);
  return x;
}

void test_basic_accuracy() {
  DeviceParams params;
  params.num_levels = 32;
  params.write_noise_frac_of_level = 0.25;
  params.read_noise_frac_of_write = 0.2;

  std::mt19937 rng(42);
  int n = 16;
  Matrix W = random_matrix(n, n, rng);
  std::vector<double> x = random_vec(n, rng);

  Crossbar xbar(n, n, params, /*seed=*/1);
  xbar.program(W);
  std::vector<double> y_analog = xbar.multiply(x);
  std::vector<double> y_ideal = ideal_matvec(W, x);
  AccuracyResult r = compare(y_analog, y_ideal);

  std::printf("  %dx%d crossbar, num_levels=%d: RMSE=%.4f, relative RMSE=%.2f%%\n", n, n, params.num_levels, r.rmse,
              r.relative_rmse * 100.0);
  require(r.relative_rmse < 0.20, "16x16 crossbar at 32 levels achieves <20% relative RMSE vs. exact digital matvec");
}

void test_precision_sweep() {
  std::mt19937 rng(7);
  int n = 16;
  Matrix W = random_matrix(n, n, rng);
  std::vector<double> x = random_vec(n, rng);
  std::vector<double> y_ideal = ideal_matvec(W, x);

  std::printf("  precision sweep (fixed %dx%d W/x, write_noise_frac_of_level=0.25):\n", n, n);
  std::vector<int> level_counts = {4, 8, 16, 32, 64};
  std::vector<double> rel_errors;
  for (int levels : level_counts) {
    DeviceParams params;
    params.num_levels = levels;
    params.write_noise_frac_of_level = 0.25;
    params.read_noise_frac_of_write = 0.2;
    Crossbar xbar(n, n, params, /*seed=*/100);
    xbar.program(W);
    std::vector<double> y_analog = xbar.multiply(x);
    AccuracyResult r = compare(y_analog, y_ideal);
    rel_errors.push_back(r.relative_rmse);
    std::printf("    num_levels=%3d -> relative RMSE=%.2f%%\n", levels, r.relative_rmse * 100.0);
  }

  require(rel_errors.front() > rel_errors.back(),
          "relative RMSE at 4 levels is worse than at 64 levels (more precision helps, measured across the full sweep)");
  int improvements = 0;
  for (size_t i = 1; i < rel_errors.size(); ++i)
    if (rel_errors[i] <= rel_errors[i - 1]) ++improvements;
  std::printf("    monotonic-or-flat improvements: %d/%zu consecutive steps\n", improvements, rel_errors.size() - 1);
  require(improvements >= static_cast<int>(rel_errors.size()) - 2,
          "accuracy improves (or holds) at nearly every step of the precision sweep, not just end-to-end");
}

void test_crossbar_size_sweep() {
  DeviceParams params;
  params.num_levels = 16;
  params.write_noise_frac_of_level = 0.25;
  params.read_noise_frac_of_write = 0.2;

  std::printf("  crossbar-size sweep (fixed num_levels=%d, fresh random W/x per size):\n", params.num_levels);
  std::vector<int> sizes = {8, 16, 32, 64};
  for (int n : sizes) {
    std::mt19937 rng(static_cast<unsigned>(1000 + n));
    Matrix W = random_matrix(n, n, rng);
    std::vector<double> x = random_vec(n, rng);
    Crossbar xbar(n, n, params, /*seed=*/static_cast<uint64_t>(2000 + n));
    xbar.program(W);
    std::vector<double> y_analog = xbar.multiply(x);
    std::vector<double> y_ideal = ideal_matvec(W, x);
    AccuracyResult r = compare(y_analog, y_ideal);
    std::printf("    %3dx%-3d -> relative RMSE=%.2f%%\n", n, n, r.relative_rmse * 100.0);
  }
  require(true, "crossbar-size sweep completed and reported (see README for the measured size-vs-accuracy relationship -- not assumed in either direction)");
}

} // namespace

int main() {
  test_basic_accuracy();
  test_precision_sweep();
  test_crossbar_size_sweep();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
