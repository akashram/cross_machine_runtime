// Verifies the state-vector simulator against known, closed-form circuit
// identities rather than just "it runs": Bell state, GHZ state, unitarity
// preservation under a random gate sequence, and quantum teleportation
// (the strongest of the four -- it exercises single-qubit gates, CNOT,
// partial measurement/projection, and classically-conditioned correction
// together, and is checked via the EXACT conditional-amplitude-ratio
// identity rather than a probabilistic sampling average).
#include "state_vector.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

using namespace quantum;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_bell_state() {
  StateVector sv(2);
  sv.h(0);
  sv.cnot(0, 1);
  double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  bool ok = std::abs(sv.amplitude(0).real() - inv_sqrt2) < 1e-9 && std::abs(sv.amplitude(0).imag()) < 1e-9 &&
            std::abs(sv.amplitude(3).real() - inv_sqrt2) < 1e-9 && std::abs(sv.amplitude(1)) < 1e-9 &&
            std::abs(sv.amplitude(2)) < 1e-9;
  std::printf("  amps: |00>=%.4f |01>=%.4f |10>=%.4f |11>=%.4f\n", sv.amplitude(0).real(), sv.amplitude(1).real(),
              sv.amplitude(2).real(), sv.amplitude(3).real());
  require(ok, "H(q0);CNOT(0,1) produces the Bell state (|00>+|11>)/sqrt(2) exactly");
}

void test_ghz_state() {
  StateVector sv(3);
  sv.h(0);
  sv.cnot(0, 1);
  sv.cnot(0, 2);
  double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  double other_mass = 0.0;
  for (std::size_t i = 0; i < sv.dim(); ++i)
    if (i != 0 && i != 7) other_mass += std::norm(sv.amplitude(i));
  bool ok = std::abs(sv.amplitude(0).real() - inv_sqrt2) < 1e-9 && std::abs(sv.amplitude(7).real() - inv_sqrt2) < 1e-9 &&
            other_mass < 1e-18;
  std::printf("  |000>=%.4f |111>=%.4f other-state probability mass=%.2e\n", sv.amplitude(0).real(),
              sv.amplitude(7).real(), other_mass);
  require(ok, "H(q0);CNOT(0,1);CNOT(0,2) produces the 3-qubit GHZ state (|000>+|111>)/sqrt(2) exactly");
}

void test_unitarity_preserved() {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> angle(0.0, 2.0 * kPi);
  std::uniform_int_distribution<int> qubit_pick(0, 3);
  StateVector sv(4);
  double worst_dev = 0.0;
  for (int trial = 0; trial < 200; ++trial) {
    int gate_kind = static_cast<int>(rng() % 6);
    int q = qubit_pick(rng);
    switch (gate_kind) {
      case 0: sv.h(q); break;
      case 1: sv.x(q); break;
      case 2: sv.rx(q, angle(rng)); break;
      case 3: sv.ry(q, angle(rng)); break;
      case 4: sv.rz(q, angle(rng)); break;
      case 5: sv.cnot(q, (q + 1) % 4); break;
    }
    worst_dev = std::max(worst_dev, std::abs(sv.norm_squared() - 1.0));
  }
  std::printf("  worst |sum|amp|^2 - 1| over 200 random gates on 4 qubits = %.3e\n", worst_dev);
  require(worst_dev < 1e-10, "norm^2 stays exactly 1 (to fp precision) after 200 random single/two-qubit unitary gates");
}

// Standard quantum teleportation: q0 carries an arbitrary (real, via RY)
// single-qubit state; q1/q2 share a Bell pair; Alice (q0,q1) performs a
// Bell-basis measurement via CNOT+H then measures classically; Bob (q2)
// applies X iff q1's outcome is 1, Z iff q0's outcome is 1. The textbook
// identity: after correction, q2's state exactly reproduces q0's original
// state, for EVERY one of the four measurement outcomes -- not just on
// average. Checked exactly (no RNG) by projecting onto all 4 outcome
// branches and comparing the resulting conditional amplitude ratio on q2
// to the original state's amplitude ratio; all gates used here (RY, H,
// CNOT, X, Z) are real-valued, so amplitudes stay real throughout and a
// direct ratio comparison is valid.
void test_teleportation() {
  const double theta = 0.7;  // arbitrary, away from 0/pi where an amplitude vanishes
  const double expected_ratio = std::tan(theta / 2.0);  // sin(theta/2) / cos(theta/2)

  StateVector prep(3);
  prep.ry(0, theta);
  prep.h(1);
  prep.cnot(1, 2);
  prep.cnot(0, 1);
  prep.h(0);

  bool all_branches_ok = true;
  for (int m0 = 0; m0 < 2; ++m0) {
    for (int m1 = 0; m1 < 2; ++m1) {
      StateVector branch = prep;  // copy: each branch projects independently
      branch.project_unnormalized(0, m0);
      branch.project_unnormalized(1, m1);
      if (m1 == 1) branch.x(2);
      if (m0 == 1) branch.z(2);

      std::size_t idx0 = static_cast<std::size_t>(m0) | (static_cast<std::size_t>(m1) << 1);
      std::size_t idx1 = idx0 | (std::size_t(1) << 2);
      cplx a0 = branch.amplitude(idx0), a1 = branch.amplitude(idx1);
      double branch_mass = std::norm(a0) + std::norm(a1);
      double ratio = a0.real() != 0.0 ? a1.real() / a0.real() : std::numeric_limits<double>::infinity();
      bool branch_ok = std::abs(ratio - expected_ratio) < 1e-9 && std::abs(a0.imag()) < 1e-12 &&
                        std::abs(a1.imag()) < 1e-12 && std::abs(branch_mass - 0.25) < 1e-9;
      std::printf("  branch (m0=%d,m1=%d): a0=%.6f a1=%.6f ratio=%.6f (expected %.6f) branch-mass=%.6f\n", m0, m1,
                  a0.real(), a1.real(), ratio, expected_ratio, branch_mass);
      all_branches_ok = all_branches_ok && branch_ok;
    }
  }
  require(all_branches_ok,
          "teleportation: after X/Z correction, q2's conditional amplitude ratio matches q0's original "
          "cos(theta/2):sin(theta/2) ratio EXACTLY on all 4 measurement branches (not just on average)");
}

}  // namespace

int main() {
  test_bell_state();
  test_ghz_state();
  test_unitarity_preserved();
  test_teleportation();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
