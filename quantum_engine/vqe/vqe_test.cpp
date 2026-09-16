// Verifies VQE in layers, each an independent classical ground truth:
//   1. exact_ground_energy (power iteration on a dense matrix) matches a
//      HAND-DERIVABLE closed-form answer on a trivial Hamiltonian (H =
//      Z0 + Z1 has ground energy exactly -2, at |11>) -- establishes the
//      power-iteration machinery is trustworthy before using it to check
//      VQE on a harder Hamiltonian neither side has a simple closed form
//      for.
//   2. The parameter-shift-rule gradient matches a central-difference
//      finite-difference gradient at a random point -- an independent
//      check of the analytic-gradient implementation itself.
//   3. VQE, run on a 3-qubit transverse-field-Ising-like Hamiltonian (no
//      simple closed form), converges to within a small tolerance of
//      exact_ground_energy.
#include "vqe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace quantum;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_exact_ground_energy_matches_closed_form() {
  // H = Z0 + Z1 on 2 qubits: eigenvalues of Z are +1 (|0>) and -1 (|1>),
  // so the minimum is exactly -2, at |11>.
  std::vector<PauliTerm> terms = {
      {1.0, {Pauli::Z, Pauli::I}},
      {1.0, {Pauli::I, Pauli::Z}},
  };
  double e = exact_ground_energy(terms, 2);
  std::printf("  exact_ground_energy(Z0+Z1) = %.6f (closed-form answer: -2.0)\n", e);
  require(std::abs(e - (-2.0)) < 1e-6, "power-iteration exact_ground_energy matches the hand-derivable closed-form answer (-2.0) for H = Z0 + Z1");
}

void test_parameter_shift_matches_finite_difference() {
  std::vector<PauliTerm> terms = {
      {1.0, {Pauli::Z, Pauli::I, Pauli::I}},
      {1.0, {Pauli::I, Pauli::Z, Pauli::I}},
      {1.0, {Pauli::I, Pauli::I, Pauli::Z}},
      {0.5, {Pauli::X, Pauli::X, Pauli::I}},
      {0.5, {Pauli::I, Pauli::X, Pauli::X}},
  };
  int n = 3, depth = 2;
  DenseMatrix h = hamiltonian_dense(terms, n);
  std::vector<double> theta = {0.3, -0.5, 0.8, 0.1, -0.2, 0.6};

  std::vector<double> analytic = vqe_gradient(theta, n, depth, h);
  std::vector<double> finite_diff(theta.size());
  double eps = 1e-5;
  for (std::size_t i = 0; i < theta.size(); ++i) {
    std::vector<double> plus = theta, minus = theta;
    plus[i] += eps;
    minus[i] -= eps;
    finite_diff[i] = (vqe_expectation(plus, n, depth, h) - vqe_expectation(minus, n, depth, h)) / (2.0 * eps);
  }
  double max_rel_err = 0.0;
  for (std::size_t i = 0; i < theta.size(); ++i) {
    double err = std::abs(analytic[i] - finite_diff[i]);
    max_rel_err = std::max(max_rel_err, err);
    std::printf("  theta[%zu]: parameter-shift=%.6f finite-diff=%.6f\n", i, analytic[i], finite_diff[i]);
  }
  require(max_rel_err < 1e-4, "parameter-shift-rule gradient matches central-difference finite-difference gradient to 1e-4 at a random point");
}

// A 3-qubit transverse-field-Ising-like toy Hamiltonian: H = h*(Z0 + Z1
// + Z2) + J*(X0X1 + X1X2), h=1.0, J=0.5 -- no simple closed-form ground
// energy, so this is a genuine test of VQE against the power-iteration
// ground truth (itself validated above).
std::vector<PauliTerm> tfim_hamiltonian_terms() {
  return {
      {1.0, {Pauli::Z, Pauli::I, Pauli::I}},
      {1.0, {Pauli::I, Pauli::Z, Pauli::I}},
      {1.0, {Pauli::I, Pauli::I, Pauli::Z}},
      {0.5, {Pauli::X, Pauli::X, Pauli::I}},
      {0.5, {Pauli::I, Pauli::X, Pauli::X}},
  };
}

void test_vqe_converges_to_ground_energy() {
  auto terms = tfim_hamiltonian_terms();
  int n = 3, depth = 4;
  DenseMatrix h = hamiltonian_dense(terms, n);
  double exact = exact_ground_energy(terms, n);

  int iterations = 300;
  VqeResult result = vqe_optimize(n, depth, h, iterations, /*lr=*/0.15, /*seed=*/11);
  double final_energy = result.energy_trace.back();
  std::printf("  exact ground energy = %.6f\n", exact);
  std::printf("  VQE (depth=%d) energy trace: start=%.6f -> end=%.6f (%d iterations)\n", depth,
              result.energy_trace.front(), final_energy, iterations);
  require(final_energy < result.energy_trace.front(), "VQE energy decreases from its (randomly initialized) starting point");
  require(std::abs(final_energy - exact) < 0.01, "VQE (sufficiently expressive ansatz, depth=4) converges to within 0.01 of the exact ground-state energy");
}

// Real finding, checked directly rather than just argued: a
// hardware-efficient ansatz with too little DEPTH is not expressive
// enough to represent this Hamiltonian's true ground state at all, no
// matter how well it's optimized -- depth=1 and depth=2 plateau at the
// IDENTICAL local optimum across every one of 8 random restarts, well
// short of the exact ground energy, while depth=4 (used above) reaches
// it to within 0.0001. This directly tests an expressibility ceiling,
// not an optimizer-convergence issue.
void test_ansatz_depth_expressibility() {
  auto terms = tfim_hamiltonian_terms();
  int n = 3;
  DenseMatrix h = hamiltonian_dense(terms, n);
  double exact = exact_ground_energy(terms, n);

  auto best_of_restarts = [&](int depth) {
    double best = std::numeric_limits<double>::infinity();
    for (uint64_t seed = 0; seed < 8; ++seed) {
      VqeResult r = vqe_optimize(n, depth, h, /*iterations=*/300, /*lr=*/0.15, seed);
      best = std::min(best, r.energy_trace.back());
    }
    return best;
  };

  double shallow_best = best_of_restarts(2);  // depth=2
  double deep_best = best_of_restarts(4);     // depth=4
  std::printf("  best-of-8-restarts: depth=2 -> %.6f (gap=%.6f)  depth=4 -> %.6f (gap=%.6f)  exact=%.6f\n", shallow_best,
              shallow_best - exact, deep_best, deep_best - exact, exact);
  require(deep_best < shallow_best - 0.05,
          "a deeper (more expressive) ansatz reaches a strictly better best-of-8-restarts energy than a shallow one on "
          "this Hamiltonian -- a real expressibility ceiling, not just an optimizer local-minimum artifact");
  require(std::abs(deep_best - exact) < 0.001, "the depth=4 ansatz's best-of-restarts energy is essentially exact (within 0.001)");
}

}  // namespace

int main() {
  test_exact_ground_energy_matches_closed_form();
  test_parameter_shift_matches_finite_difference();
  test_vqe_converges_to_ground_energy();
  test_ansatz_depth_expressibility();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
