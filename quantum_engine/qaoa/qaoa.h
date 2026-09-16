// Phase 20 step 6: QAOA (Quantum Approximate Optimization Algorithm) on a
// small MaxCut instance, with an HONEST comparison against a classical
// brute-force baseline -- in the spirit of Phase 12c's "does the
// sophisticated method actually beat the simple baseline at this scale"
// findings, not an assumed quantum-advantage narrative.
#pragma once

#include "../state_vector/state_vector.h"

#include <algorithm>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace quantum {

using Edge = std::pair<int, int>;
using Graph = std::vector<Edge>;

// Counts cut edges for a given bit assignment (bit i = which side node i
// is on).
inline int cut_value(const Graph &edges, std::size_t assignment) {
  int cut = 0;
  for (const auto &[i, j] : edges) {
    bool bi = (assignment >> i) & 1;
    bool bj = (assignment >> j) & 1;
    if (bi != bj) ++cut;
  }
  return cut;
}

// Exhaustive classical baseline: try every 2^n bipartition, return the
// best cut value found and the assignment that achieves it.
inline std::pair<int, std::size_t> classical_max_cut_bruteforce(int n, const Graph &edges) {
  std::size_t best_assignment = 0;
  int best = -1;
  for (std::size_t a = 0; a < (std::size_t(1) << n); ++a) {
    int c = cut_value(edges, a);
    if (c > best) { best = c; best_assignment = a; }
  }
  return {best, best_assignment};
}

// Implements exp(-i*(angle/2)*Z_i*Z_j) via the standard CNOT-RZ-CNOT
// identity (conjugating an RZ by CNOTs turns a single-qubit Z rotation
// into a two-qubit ZZ rotation, since CNOT(i,j) maps Z_j -> Z_i*Z_j).
// Verified directly against a brute-force diagonal-phase check in
// qaoa_test.cpp before being trusted inside the full QAOA circuit.
inline void apply_zz(StateVector &sv, int i, int j, double angle) {
  sv.cnot(i, j);
  sv.rz(j, angle);
  sv.cnot(i, j);
}

// The standard QAOA ansatz: H on every qubit (uniform superposition),
// then p layers of (cost unitary exp(-i*gamma*C); mixer unitary
// exp(-i*beta*B)). C = sum_edges 0.5*(I - Z_i Z_j) (MaxCut cost
// Hamiltonian); since -gamma*C's constant/identity term is a global
// phase (irrelevant to any expectation value) the cost unitary per edge
// reduces to exp(i*(gamma/2)*Z_i*Z_j) = apply_zz(..., angle=-gamma).
// B = sum_i X_i (mixer Hamiltonian); since each X_i acts on a different
// qubit, exp(-i*beta*B) factors into RX(2*beta) on every qubit
// (RX(theta) = exp(-i*theta/2*X) in this simulator's convention).
// `theta` layout: [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}].
inline StateVector qaoa_circuit(const std::vector<double> &theta, int n, const Graph &edges, int p) {
  StateVector sv(n);
  for (int q = 0; q < n; ++q) sv.h(q);
  for (int layer = 0; layer < p; ++layer) {
    double gamma = theta[static_cast<std::size_t>(layer)];
    double beta = theta[static_cast<std::size_t>(p + layer)];
    for (const auto &[i, j] : edges) apply_zz(sv, i, j, -gamma);
    for (int q = 0; q < n; ++q) sv.rx(q, 2.0 * beta);
  }
  return sv;
}

// Exact expected cut value <C> = sum_edges 0.5*(1 - <Z_i Z_j>), computed
// directly from the state vector's amplitudes (no shot-based sampling --
// the simulator has full amplitude access, unlike real hardware; see
// README for the disclosed simplification).
inline double qaoa_expected_cut(const std::vector<double> &theta, int n, const Graph &edges, int p) {
  StateVector sv = qaoa_circuit(theta, n, edges, p);
  double expected = 0.0;
  for (const auto &[i, j] : edges) {
    double zz = 0.0;
    for (std::size_t basis = 0; basis < sv.dim(); ++basis) {
      double pr = std::norm(sv.amplitude(basis));
      if (pr == 0.0) continue;
      int zi = ((basis >> i) & 1) ? -1 : 1;
      int zj = ((basis >> j) & 1) ? -1 : 1;
      zz += pr * zi * zj;
    }
    expected += 0.5 * (1.0 - zz);
  }
  return expected;
}

struct QaoaResult {
  std::vector<double> theta;
  std::vector<double> cut_trace;
};

// Gradient ASCENT (maximizing expected cut) via central-difference
// finite differences. Finite differences, not the exact parameter-shift
// rule used by vqe.h: gamma_l and beta_l each appear in MULTIPLE gate
// occurrences per layer (one apply_zz call per edge, one rx call per
// qubit) sharing the SAME parameter, so the simple two-point
// parameter-shift rule doesn't directly apply to the whole layer as one
// gate -- see README for the full reasoning (this repo's established
// finite-difference-GD pattern, used elsewhere e.g. sciml/ssm_layer and
// sciml/mup_scaling, is the pragmatic, correct-by-construction choice
// here).
inline QaoaResult qaoa_optimize(int n, const Graph &edges, int p, int iterations, double lr, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.1, 1.0);
  std::vector<double> theta(static_cast<std::size_t>(2 * p));
  for (auto &t : theta) t = u(rng);

  QaoaResult result;
  double eps = 1e-4;
  for (int it = 0; it < iterations; ++it) {
    double c = qaoa_expected_cut(theta, n, edges, p);
    result.cut_trace.push_back(c);
    std::vector<double> grad(theta.size());
    for (std::size_t k = 0; k < theta.size(); ++k) {
      std::vector<double> plus = theta, minus = theta;
      plus[k] += eps;
      minus[k] -= eps;
      grad[k] = (qaoa_expected_cut(plus, n, edges, p) - qaoa_expected_cut(minus, n, edges, p)) / (2.0 * eps);
    }
    for (std::size_t k = 0; k < theta.size(); ++k) theta[k] += lr * grad[k];  // ascent
  }
  result.theta = theta;
  return result;
}

}  // namespace quantum
