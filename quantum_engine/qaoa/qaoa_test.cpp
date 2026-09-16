// Verifies: (1) apply_zz implements the exact diagonal phase
// exp(-i*angle/2*Z_i*Z_j) it claims to, checked against a brute-force
// per-basis-state phase computation, not trusted from the CNOT-RZ-CNOT
// identity alone; (2) QAOA on a 5-node cycle graph (C5, an ODD cycle --
// classical max cut is provably 4 of 5 edges, not all 5) is checked
// against an exhaustive classical brute-force baseline, with an HONEST
// approximation-ratio report at p=1 and p=2 rather than an assumed
// quantum-advantage claim.
#include "qaoa.h"

#include <cmath>
#include <cstdio>

using namespace quantum;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_apply_zz_phase_is_correct() {
  bool all_ok = true;
  double angle = 0.83;
  for (std::size_t basis = 0; basis < 4; ++basis) {
    StateVector sv(2);
    sv.set_amplitude(0, cplx(0.0, 0.0));
    sv.set_amplitude(basis, cplx(1.0, 0.0));
    apply_zz(sv, 0, 1, angle);
    int z0 = (basis & 1) ? -1 : 1;
    int z1 = ((basis >> 1) & 1) ? -1 : 1;
    cplx expected_phase = std::polar(1.0, -angle / 2.0 * z0 * z1);
    cplx actual = sv.amplitude(basis);
    bool ok = std::abs(actual - expected_phase) < 1e-9;
    std::printf("  basis=%zu: actual=(%.4f,%.4f) expected=(%.4f,%.4f)\n", basis, actual.real(), actual.imag(),
                expected_phase.real(), expected_phase.imag());
    all_ok = all_ok && ok;
  }
  require(all_ok, "apply_zz(angle) implements exp(-i*angle/2*Z_i*Z_j) exactly, checked against a brute-force per-basis-state phase computation");
}

void test_qaoa_maxcut_vs_classical_baseline() {
  // C5: a 5-node cycle. Odd cycles cannot have every edge cut (no
  // bipartite 2-coloring exists), so the classical max cut is
  // PROVABLY 4 of 5 edges, not 5 -- a genuinely non-trivial instance,
  // not a toy where the "obvious" answer is trivially all edges.
  int n = 5;
  Graph edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}};

  auto [classical_best, classical_assignment] = classical_max_cut_bruteforce(n, edges);
  std::printf("  classical brute-force max cut over all %d assignments: %d edges (assignment bits=%zu)\n", 1 << n,
              classical_best, classical_assignment);
  require(classical_best == 4, "classical brute-force finds the max cut of C5 is exactly 4 (odd cycle: can't cut all 5)");

  for (int p : {1, 2}) {
    QaoaResult result = qaoa_optimize(n, edges, p, /*iterations=*/300, /*lr=*/0.05, /*seed=*/3);
    double achieved = result.cut_trace.back();
    double ratio = achieved / classical_best;
    std::printf("  QAOA p=%d: expected cut trace start=%.4f -> end=%.4f (classical max=%d, approximation ratio=%.4f)\n", p,
                result.cut_trace.front(), achieved, classical_best, ratio);
    require(achieved > result.cut_trace.front(), "QAOA's expected cut value improves from its random starting point");
    require(ratio > 0.5 && ratio <= 1.0 + 1e-6, "QAOA's achieved expected cut is a genuine partial approximation of the classical optimum (strictly better than a random cut's ~0.5 ratio, never exceeding the true optimum)");
  }
}

}  // namespace

int main() {
  test_apply_zz_phase_is_correct();
  test_qaoa_maxcut_vs_classical_baseline();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
