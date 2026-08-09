// device_model_test.cpp -- verifies the four modeled non-idealities behave
// the way real RRAM-class devices are reported to (Yu 2018), not just that
// the code runs:
//   1. Read noise is measurably smaller than write noise (read() perturbs
//      the cell far less than write()) -- checked by comparing the spread
//      of repeated reads of one write vs. the spread across repeated
//      independent writes to the same level.
//   2. Level round-trip: writing level L then reading immediately recovers
//      L via nearest-level classification with high accuracy at a modest
//      level count -- basic correctness, and the accuracy this degrades
//      FROM as num_levels grows is exactly step 2's crossbar precision
//      story.
//   3. Drift: conductance measured a long time after write is lower than
//      immediately after write (nu > 0 => monotone decay), averaged over
//      many trials to separate the drift signal from noise.
//   4. Endurance: cells written far below rated_endurance_cycles almost
//      never end up stuck; cells written far beyond it do, measurably.
#include "device_model.h"

#include <cmath>
#include <cstdio>
#include <numeric>

using namespace analog;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

double stddev(const std::vector<double> &v) {
  double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  double sq = 0.0;
  for (double x : v) sq += (x - mean) * (x - mean);
  return std::sqrt(sq / v.size());
}

void test_read_noise_smaller_than_write_noise() {
  DeviceParams p;
  p.num_levels = 16;

  // Spread across repeated INDEPENDENT WRITES of the same level, each read
  // once immediately (elapsed=0 -> t_eff clamps to drift_t_ref, negligible
  // drift at that scale).
  std::vector<double> write_spread;
  for (int trial = 0; trial < 500; ++trial) {
    ConductanceCell cell(p, /*seed=*/static_cast<uint64_t>(1000 + trial));
    cell.write(8);
    write_spread.push_back(cell.read(0.0));
  }

  // Spread across repeated READS of a SINGLE write.
  ConductanceCell fixed_cell(p, /*seed=*/7);
  fixed_cell.write(8);
  std::vector<double> read_spread;
  for (int trial = 0; trial < 500; ++trial) read_spread.push_back(fixed_cell.read(0.0));

  double sd_write = stddev(write_spread);
  double sd_read = stddev(read_spread);
  std::printf("  stddev across independent writes = %.4f | stddev across repeated reads of one write = %.4f\n",
              sd_write, sd_read);
  require(sd_read < sd_write, "read() perturbs a cell's measured conductance far less than write() does (read noise < write noise, as physically expected)");
}

void test_level_roundtrip_accuracy() {
  DeviceParams p;
  p.num_levels = 8; // coarse, well-separated levels
  p.write_noise_frac_of_level = 0.25;

  int correct = 0, total = 0;
  for (int level = 0; level < p.num_levels; ++level) {
    for (int trial = 0; trial < 50; ++trial) {
      ConductanceCell cell(p, /*seed=*/static_cast<uint64_t>(level * 1000 + trial));
      cell.write(level);
      double measured = cell.read(0.0);
      // Nearest-level classification.
      double step = cell.level_step();
      int recovered = static_cast<int>(std::round((measured - p.g_min) / step));
      recovered = std::clamp(recovered, 0, p.num_levels - 1);
      if (recovered == level) ++correct;
      ++total;
    }
  }
  double accuracy = static_cast<double>(correct) / total;
  std::printf("  nearest-level round-trip accuracy at num_levels=%d: %.1f%% (%d/%d)\n", p.num_levels,
              accuracy * 100.0, correct, total);
  require(accuracy > 0.95, "writing a level then reading immediately recovers it via nearest-level classification with >95% accuracy at 8 well-separated levels");
}

void test_drift_reduces_conductance_over_time() {
  DeviceParams p;
  p.num_levels = 16;
  p.drift_nu = 0.08;
  p.drift_t_ref = 1.0;

  const int trials = 300;
  std::vector<double> early, late;
  for (int trial = 0; trial < trials; ++trial) {
    ConductanceCell cell(p, /*seed=*/static_cast<uint64_t>(5000 + trial));
    cell.write(12);
    early.push_back(cell.read(p.drift_t_ref)); // no drift yet
  }
  for (int trial = 0; trial < trials; ++trial) {
    ConductanceCell cell(p, /*seed=*/static_cast<uint64_t>(5000 + trial));
    cell.write(12);
    late.push_back(cell.read(p.drift_t_ref * 1.0e6)); // long after write
  }
  double mean_early = std::accumulate(early.begin(), early.end(), 0.0) / early.size();
  double mean_late = std::accumulate(late.begin(), late.end(), 0.0) / late.size();
  std::printf("  mean conductance: t=t_ref -> %.4f | t=1e6*t_ref -> %.4f (nu=%.2f)\n", mean_early, mean_late,
              p.drift_nu);
  require(mean_late < mean_early, "conductance drifts DOWNWARD over time under the G(t) = G0*(t/t_ref)^(-nu) power-law model (nu > 0)");

  DeviceParams p_no_drift = p;
  p_no_drift.drift_nu = 0.0;
  std::vector<double> no_drift_late;
  for (int trial = 0; trial < trials; ++trial) {
    ConductanceCell cell(p_no_drift, /*seed=*/static_cast<uint64_t>(5000 + trial));
    cell.write(12);
    no_drift_late.push_back(cell.read(p.drift_t_ref * 1.0e6));
  }
  double mean_no_drift_late = std::accumulate(no_drift_late.begin(), no_drift_late.end(), 0.0) / no_drift_late.size();
  std::printf("  with drift_nu=0: mean conductance at t=1e6*t_ref -> %.4f (should match early-time mean, no decay)\n",
              mean_no_drift_late);
  require(std::abs(mean_no_drift_late - mean_early) < 0.5, "drift_nu=0 leaves conductance essentially unchanged over the same time span (isolates drift as the cause above, not some other time-dependent effect)");
}

void test_endurance_stuck_fraction() {
  DeviceParams p;
  p.num_levels = 16;
  p.rated_endurance_cycles = 1000.0;
  p.stuck_sharpness = 4.0;

  auto stuck_fraction_after = [&](uint64_t num_writes) {
    const int cells = 200;
    int stuck = 0;
    for (int c = 0; c < cells; ++c) {
      ConductanceCell cell(p, /*seed=*/static_cast<uint64_t>(9000 + c));
      for (uint64_t w = 0; w < num_writes; ++w) cell.write(static_cast<int>(w % static_cast<uint64_t>(p.num_levels)));
      if (cell.is_stuck()) ++stuck;
    }
    return static_cast<double>(stuck) / cells;
  };

  double frac_below = stuck_fraction_after(100);   // far below rated endurance
  double frac_above = stuck_fraction_after(100000); // far above rated endurance
  std::printf("  stuck fraction after 100 writes (100x below rated endurance) = %.3f\n", frac_below);
  std::printf("  stuck fraction after 100000 writes (100x above rated endurance) = %.3f\n", frac_above);
  require(frac_below < 0.05, "cells written far below their rated endurance are almost never stuck");
  require(frac_above > 0.90, "cells written far beyond their rated endurance become stuck with high probability");
}

} // namespace

int main() {
  test_read_noise_smaller_than_write_noise();
  test_level_roundtrip_accuracy();
  test_drift_reduces_conductance_over_time();
  test_endurance_stuck_fraction();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
