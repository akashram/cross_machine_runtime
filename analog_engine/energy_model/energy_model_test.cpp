// energy_model_test.cpp -- prints the full pJ/MAC comparison and checks
// two real structural facts about the model:
//   1. ADC overhead dominates: the "pure compute" analog figure is orders
//      of magnitude cheaper than the "realistic" (ADC-inclusive) figure at
//      any usable precision -- the real, well-documented reason naive
//      "analog compute is free" claims are misleading in isolation.
//   2. Realistic analog energy grows monotonically with num_levels (more
//      bits of ADC resolution costs more) -- connecting directly to step
//      2's finding that num_levels is what actually drives MAC accuracy:
//      precision has a real energy cost, not a free lunch.
// The CPU/GPU/NPU vs. analog comparison itself is reported, not asserted
// in a fixed direction -- see README for the actual (somewhat
// counter-intuitive) result.
#include "energy_model.h"

#include <cstdio>

using namespace analog;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

} // namespace

int main() {
  std::printf("  digital devices (pJ/MAC, derived from npu_engine/cost_model's TOPS/W constants):\n");
  for (const auto &d : digital_devices())
    std::printf("    %-28s %.3f TOPS/W -> %.4f pJ/MAC\n", d.name.c_str(), d.tops_per_watt,
                digital_pj_per_mac(d.tops_per_watt));

  AnalogEstimate a = analog_estimate();
  std::printf("\n  analog crossbar (illustrative, literature order-of-magnitude, no fab access):\n");
  std::printf("    pure compute (Ohm's-law MAC alone): %.4f pJ/MAC\n", a.pure_compute_pj_per_mac);
  std::printf("    realistic (ADC-inclusive) by precision:\n");
  std::vector<int> level_counts = {4, 8, 16, 32, 64};
  std::vector<double> realistic;
  for (int levels : level_counts) {
    double pj = analog_realistic_pj_per_mac(a, levels);
    realistic.push_back(pj);
    std::printf("      num_levels=%3d (%.0f-bit): %.4f pJ/MAC\n", levels, std::log2(static_cast<double>(levels)), pj);
  }

  require(a.pure_compute_pj_per_mac < realistic.front() / 10.0,
          "pure-compute analog MAC energy is more than 10x cheaper than the ADC-inclusive realistic figure, even at the COARSEST precision -- ADC overhead dominates, the real reason isolated 'analog is free' claims mislead");

  bool monotonic = true;
  for (size_t i = 1; i < realistic.size(); ++i)
    if (realistic[i] <= realistic[i - 1]) monotonic = false;
  require(monotonic, "realistic (ADC-inclusive) analog MAC energy strictly increases with num_levels -- precision has a real energy cost, connecting directly to step 2's finding that precision drives accuracy");

  double npu_pj = digital_pj_per_mac(digital_devices()[2].tops_per_watt);
  double gpu_pj = digital_pj_per_mac(digital_devices()[1].tops_per_watt);
  double cpu_pj = digital_pj_per_mac(digital_devices()[0].tops_per_watt);
  double analog_at_32_levels = analog_realistic_pj_per_mac(a, 32);
  std::printf("\n  at num_levels=32 (5-bit, step 2's representative precision): realistic analog=%.4f pJ/MAC vs. NPU=%.4f, GPU=%.4f, CPU=%.4f\n",
              analog_at_32_levels, npu_pj, gpu_pj, cpu_pj);
  require(analog_at_32_levels < gpu_pj && analog_at_32_levels < cpu_pj,
          "realistic (ADC-inclusive) analog crossbar beats both GPU and CPU on pJ/MAC at 32-level precision -- a real margin, not just the misleading pure-compute number");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
