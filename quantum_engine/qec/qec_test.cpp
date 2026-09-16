// Verifies the 3-qubit bit-flip code: (1) with zero noise it's a
// perfect no-op; (2) against pure bit-flip noise (the channel it's
// designed for), measured logical error rate is far below physical
// error rate at small p, and matches the analytic formula
// 3p^2 - 2p^3 (probability of >=2 of 3 independent errors) closely;
// (3) against general depolarizing noise (X, Y, AND Z errors), the code
// protects much less well -- an honest, disclosed limitation, not
// hidden.
#include "qec.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace quantum;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_zero_noise_is_perfect() {
  std::mt19937_64 rng(1);
  double rate = qec_logical_error_rate(0.7, /*p_phys=*/0.0, qec_inject_bitflip_errors, /*trials=*/50, /*seed=*/1);
  std::printf("  logical error rate at p_phys=0.0 = %.4f\n", rate);
  require(rate == 0.0, "zero physical error rate gives zero logical error rate (encode/syndrome/correct/decode is a perfect no-op with no injected error)");
}

void test_bitflip_protection() {
  bool all_ok = true;
  std::vector<double> ps = {0.01, 0.05, 0.1, 0.2};
  for (double p : ps) {
    double measured = qec_logical_error_rate(0.7, p, qec_inject_bitflip_errors, /*trials=*/2000, /*seed=*/42);
    double analytic = 3.0 * p * p - 2.0 * p * p * p;  // P(>=2 of 3 independent errors)
    std::printf("  p_phys=%.2f: measured logical error rate=%.4f analytic(3p^2-2p^3)=%.4f physical(p)=%.4f\n", p, measured,
                analytic, p);
    bool close_to_analytic = std::abs(measured - analytic) < 0.03;  // finite-sample tolerance at 2000 trials
    bool beats_physical = measured < p;
    all_ok = all_ok && close_to_analytic && beats_physical;
  }
  require(all_ok, "against pure bit-flip noise, measured logical error rate matches the analytic 3p^2-2p^3 formula (within finite-sample tolerance) and beats the uncorrected physical error rate at every p tested");
}

void test_depolarizing_protection_is_weaker() {
  double p = 0.05;
  double bitflip_rate = qec_logical_error_rate(0.7, p, qec_inject_bitflip_errors, /*trials=*/2000, /*seed=*/99);
  double depolarizing_rate = qec_logical_error_rate(0.7, p, qec_inject_depolarizing_errors, /*trials=*/2000, /*seed=*/99);
  std::printf("  at p=%.2f: logical error rate under pure bit-flip noise=%.4f vs. under general depolarizing noise=%.4f\n", p,
              bitflip_rate, depolarizing_rate);
  require(depolarizing_rate > bitflip_rate, "the bit-flip code protects noticeably LESS well against general depolarizing noise (X+Y+Z errors) than against the pure bit-flip noise it's designed for -- an honest, expected limitation of a code that only targets one error type");
}

}  // namespace

int main() {
  test_zero_noise_is_perfect();
  test_bitflip_protection();
  test_depolarizing_protection_is_weaker();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
