// pipeline_1f1b_test.cpp — three checks against the discrete-event
// simulation in pipeline_schedule.h:
//  1. GPipe's simulated bubble fraction matches the closed-form formula
//     (p-1)/(m+p-1), confirming the simulator's dependency handling is
//     correct against known-correct theory.
//  2. 1F1B's bubble fraction matches GPipe's (the two are NOT supposed to
//     differ — this is the real, published result: 1F1B's win is memory,
//     not bubble time — see pipeline_schedule.h).
//  3. 1F1B's peak in-flight activations per stage is bounded (~p), while
//     GPipe's equals m at every stage — the actual memory difference this
//     step exists to demonstrate.
// Also checks PLAN.md's stated definition of done: bubble fraction < 5%
// with sufficient microbatches.
#include "pipeline_schedule.h"

#include <cmath>
#include <cstdio>

using namespace distributed_training;

namespace {

bool check_configuration(int p, int m, double t_f, double t_b) {
  auto gpipe = simulate(gpipe_schedule(p, m), t_f, t_b);
  auto ofob = simulate(one_f_one_b_schedule(p, m), t_f, t_b);

  double expected_bubble = static_cast<double>(p - 1) / static_cast<double>(m + p - 1);

  bool ok = true;
  double gpipe_formula_err = std::abs(gpipe.bubble_fraction - expected_bubble);
  if (gpipe_formula_err > 1e-6) ok = false;

  double bubble_diff = std::abs(gpipe.bubble_fraction - ofob.bubble_fraction);
  if (bubble_diff > 1e-6) ok = false;

  int gpipe_max_peak = 0, ofob_max_peak = 0;
  for (int v : gpipe.peak_in_flight_per_stage) gpipe_max_peak = std::max(gpipe_max_peak, v);
  for (int v : ofob.peak_in_flight_per_stage) ofob_max_peak = std::max(ofob_max_peak, v);
  bool gpipe_peak_is_m = (gpipe_max_peak == m);
  bool ofob_peak_bounded = (ofob_max_peak <= p);
  if (!gpipe_peak_is_m || !ofob_peak_bounded) ok = false;

  std::printf(
      "  p=%2d m=%3d: GPipe bubble=%.4f (formula %.4f, err %.2e) | 1F1B bubble=%.4f (diff %.2e) | "
      "peak in-flight: GPipe=%d (== m? %s) 1F1B=%d (<= p? %s): %s\n",
      p, m, gpipe.bubble_fraction, expected_bubble, gpipe_formula_err, ofob.bubble_fraction, bubble_diff,
      gpipe_max_peak, gpipe_peak_is_m ? "yes" : "no", ofob_max_peak, ofob_peak_bounded ? "yes" : "no",
      ok ? "PASS" : "FAIL");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  std::printf("test 1-3 (bubble formula match, 1F1B==GPipe bubble, peak in-flight memory):\n");
  ok = check_configuration(4, 10, 1.0, 2.0) && ok;
  ok = check_configuration(8, 32, 1.0, 2.0) && ok;
  ok = check_configuration(2, 6, 1.0, 1.0) && ok; // t_f==t_b edge case
  ok = check_configuration(6, 6, 1.0, 2.0) && ok; // m == p edge case (num_warmup saturates)

  std::printf("\ntest 4 (PLAN.md definition of done: bubble < 5%% with sufficient microbatches):\n");
  auto result = simulate(one_f_one_b_schedule(8, 200), 1.0, 2.0);
  bool dod_ok = result.bubble_fraction < 0.05;
  std::printf("  p=8 m=200: 1F1B bubble fraction = %.4f (%.2f%%): %s\n", result.bubble_fraction,
              result.bubble_fraction * 100.0, dod_ok ? "PASS" : "FAIL");
  ok = dod_ok && ok;

  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
