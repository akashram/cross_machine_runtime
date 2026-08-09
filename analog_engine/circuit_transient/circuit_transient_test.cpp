// circuit_transient_test.cpp -- three real checks:
//   1. Step-response formula sanity: at t=tau, a first-order RC step
//      response must reach exactly (1 - 1/e) ~= 63.2% of its final value
//      -- textbook circuit theory, checked directly in code.
//   2. Self-consistency: does step_response() at the CLOSED-FORM
//      settling_time_seconds() actually land at the target threshold
//      fraction? (Same style of self-consistency check
//      fpga_engine/thermal_router_sim.cpp uses for its own RC model.)
//   3. tau scales QUADRATICALLY with crossbar size (doubling size ->
//      ~4x tau) -- the real distributed-RC-line (Elmore) result, not
//      assumed, checked against this step's own formula's output.
#include "circuit_transient.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace analog;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

} // namespace

int main() {
  RCLineParams rc;

  // Check 1: step response at t=tau.
  double tau_test = tau_seconds(rc, 32);
  double v_at_tau = step_response(1.0, tau_test, tau_test);
  double expected = 1.0 - std::exp(-1.0); // ~0.6321
  std::printf("  V(t=tau)/V_final = %.6f (theory: 1 - 1/e = %.6f)\n", v_at_tau, expected);
  require(std::abs(v_at_tau - expected) < 1e-9, "step_response at t=tau matches the textbook 1-1/e fraction exactly");

  // Check 2: self-consistency of the closed-form settling time.
  double t_settle = settling_time_seconds(tau_test, 0.99);
  double v_at_settle = step_response(1.0, tau_test, t_settle);
  std::printf("  settling_time_seconds(0.99 threshold) = %.4e s -> step_response there = %.6f (target: 0.99)\n",
              t_settle, v_at_settle);
  require(std::abs(v_at_settle - 0.99) < 1e-9, "the closed-form settling time, fed back into step_response, lands exactly at the target threshold fraction");

  // Check 3: quadratic scaling of tau with crossbar size.
  double tau_32 = tau_seconds(rc, 32);
  double tau_64 = tau_seconds(rc, 64);
  double ratio = tau_64 / tau_32;
  std::printf("  tau(32)=%.4e s, tau(64)=%.4e s, ratio=%.4f (theory: exactly 4.0, quadratic in crossbar size)\n",
              tau_32, tau_64, ratio);
  require(std::abs(ratio - 4.0) < 1e-9, "tau scales EXACTLY quadratically with crossbar size (doubling size -> 4x tau), the real distributed-RC-line result");

  // Real numbers across step 2's actual crossbar sizes.
  std::printf("\n  crossbar bitline RC step response by size (r_per_cell=%.0f ohm, c_per_cell=%.1e F):\n",
              rc.r_per_cell_ohm, rc.c_per_cell_farad);
  std::printf("  %8s %14s %18s %16s\n", "size", "tau", "settling_time(99%)", "bandwidth");
  std::vector<int> sizes = {8, 16, 32, 64, 128};
  for (int n : sizes) {
    double tau = tau_seconds(rc, n);
    double settle = settling_time_seconds(tau, 0.99);
    double bw = bandwidth_hz(tau);
    std::printf("  %8d %11.3e s %15.3e s %13.3e Hz\n", n, tau, settle, bw);
  }

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
