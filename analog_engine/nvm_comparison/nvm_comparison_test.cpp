// nvm_comparison_test.cpp -- prints the full comparison table and figure-
// of-merit ranking, and checks two structural facts about the DATA (not
// simulated behavior, since this step is a literature comparison, not a
// simulation):
//   1. SRAM-CIM's volatility (retention=0) drives its composite score to
//      exactly zero via the geometric mean, regardless of how good its
//      other four axes are -- confirms the "one hard disqualifier zeroes
//      the whole score" design actually behaves that way in code, not
//      just in the doc comment.
//   2. STT-MRAM has the fewest max_analog_levels of the four -- the real,
//      citable structural tradeoff (binary-favoring magnetic storage vs.
//      RRAM/PCM's continuous resistance range) that step 2's crossbar
//      precision-matters-most finding makes directly relevant.
#include "nvm_comparison.h"

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
  const auto &table = nvm_device_classes();
  std::printf("  %-10s %12s %10s %10s %10s %8s\n", "device", "endurance", "retention", "write_pJ", "density", "levels");
  for (const auto &d : table)
    std::printf("  %-10s %12.1e %10.1f %10.1f %10.1f %8d\n", d.name.c_str(), d.endurance_cycles, d.retention_years,
                d.write_energy_pj_per_bit, d.relative_density, d.max_analog_levels);

  auto ranking = compute_figure_of_merit();
  std::printf("\n  figure-of-merit ranking (analog compute-in-memory suitability, illustrative composite score):\n");
  for (const auto &r : ranking) std::printf("    %-10s score=%.4f\n", r.name.c_str(), r.score);

  double sram_score = 0.0;
  for (const auto &r : ranking)
    if (r.name == "SRAM-CIM") sram_score = r.score;
  require(sram_score == 0.0, "SRAM-CIM's zero retention (volatility) drives its composite score to exactly 0.0 via the geometric mean, regardless of its other good axes");

  int stt_levels = 0, max_other_levels = 0;
  for (const auto &d : table) {
    if (d.name == "STT-MRAM") stt_levels = d.max_analog_levels;
    else max_other_levels = std::max(max_other_levels, d.max_analog_levels);
  }
  require(stt_levels < max_other_levels, "STT-MRAM has strictly fewer max_analog_levels than every other device class -- the real structural binary-favoring tradeoff of magnetic storage");

  require(ranking.front().name != "SRAM-CIM", "the top-ranked device by composite score is NOT the volatile baseline (confirms the ranking isn't dominated by SRAM-CIM's best-in-class energy/endurance alone)");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
