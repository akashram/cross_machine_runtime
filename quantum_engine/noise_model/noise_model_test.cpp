// Verifies: (1) zero noise gives fidelity exactly 1 (sanity check the
// Monte Carlo machinery isn't perturbing anything when it shouldn't);
// (2) fidelity decreases monotonically as circuit DEPTH grows, at fixed
// noise strength; (3) fidelity decreases monotonically as NOISE STRENGTH
// grows, at fixed depth; both checked separately for depolarizing and
// amplitude-damping noise.
#include "noise_model.h"

#include <cstdio>
#include <string>

using namespace quantum;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_zero_noise_is_exact() {
  NoiseParams np;  // both zero
  double f = average_fidelity(4, 5, np, /*trials=*/20, /*seed=*/1);
  std::printf("  fidelity at zero noise = %.10f\n", f);
  require(std::abs(f - 1.0) < 1e-9, "zero depolarizing_p and zero amplitude_damping_gamma give fidelity exactly 1.0 (Monte Carlo machinery introduces no spurious perturbation)");
}

void test_fidelity_decreases_with_depth(const char *label, NoiseParams np) {
  std::vector<int> depths = {1, 2, 4, 8, 16};
  std::vector<double> fids;
  for (int d : depths) fids.push_back(average_fidelity(4, d, np, /*trials=*/400, /*seed=*/100));
  std::printf("  [%s] depth-> ", label);
  for (std::size_t i = 0; i < depths.size(); ++i) std::printf("%d:%.4f ", depths[i], fids[i]);
  std::printf("\n");
  bool monotone = true;
  for (std::size_t i = 1; i < fids.size(); ++i) monotone = monotone && fids[i] <= fids[i - 1] + 1e-6;
  require(monotone, (std::string("[") + label + "] fidelity is non-increasing as circuit depth grows at fixed noise strength").c_str());
}

void test_fidelity_decreases_with_noise_strength(const char *label, bool depolarizing) {
  std::vector<double> strengths = {0.0, 0.02, 0.05, 0.1, 0.2};
  std::vector<double> fids;
  for (double s : strengths) {
    NoiseParams np;
    if (depolarizing) np.depolarizing_p = s; else np.amplitude_damping_gamma = s;
    fids.push_back(average_fidelity(4, 6, np, /*trials=*/400, /*seed=*/200));
  }
  std::printf("  [%s] strength-> ", label);
  for (std::size_t i = 0; i < strengths.size(); ++i) std::printf("%.2f:%.4f ", strengths[i], fids[i]);
  std::printf("\n");
  bool monotone = true;
  for (std::size_t i = 1; i < fids.size(); ++i) monotone = monotone && fids[i] <= fids[i - 1] + 1e-6;
  require(monotone, (std::string("[") + label + "] fidelity is non-increasing as noise strength grows at fixed depth=6").c_str());
}

}  // namespace

int main() {
  test_zero_noise_is_exact();

  NoiseParams depol; depol.depolarizing_p = 0.03;
  test_fidelity_decreases_with_depth("depolarizing p=0.03", depol);

  NoiseParams damp; damp.amplitude_damping_gamma = 0.03;
  test_fidelity_decreases_with_depth("amplitude-damping gamma=0.03", damp);

  test_fidelity_decreases_with_noise_strength("depolarizing", /*depolarizing=*/true);
  test_fidelity_decreases_with_noise_strength("amplitude-damping", /*depolarizing=*/false);

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
