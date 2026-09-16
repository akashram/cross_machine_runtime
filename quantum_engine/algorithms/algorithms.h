// Phase 20 step 2: canonical algorithms on state_vector's simulator --
// Deutsch-Jozsa, Grover's search, and the Quantum Fourier Transform. Each
// is checked against an independent classical ground truth (see
// algorithms_test.cpp), not just "the circuit runs and produces output".
#pragma once

#include "../state_vector/state_vector.h"

#include <cmath>
#include <complex>
#include <functional>
#include <vector>

namespace quantum {

// ---------------------------------------------------------------------
// Deutsch-Jozsa: determine whether an n-bit boolean function f is
// constant or balanced with a SINGLE oracle query (vs. up to 2^(n-1)+1
// classical queries in the worst case). `oracle` implements the
// standard phase-kickback unitary U_f: |x>|y> -> |x>|y XOR f(x)>, acting
// on qubits [0, n) as x and qubit `ancilla` (== n here) as y.
using OracleFn = std::function<void(StateVector &, int n_input_qubits, int ancilla_qubit)>;

// Returns true iff f is constant (deterministic given an ideal oracle --
// the post-circuit probability of measuring all-zero on the input
// register is EXACTLY 1.0 for constant f and EXACTLY 0.0 for balanced f,
// so no sampling/measurement is needed to get the answer).
inline bool deutsch_jozsa(int n, const OracleFn &oracle) {
  StateVector sv(n + 1);
  int ancilla = n;
  sv.x(ancilla);
  for (int q = 0; q <= n; ++q) sv.h(q);
  oracle(sv, n, ancilla);
  for (int q = 0; q < n; ++q) sv.h(q);

  double prob_all_zero_input = 0.0;
  std::size_t input_mask = (std::size_t(1) << n) - 1;
  for (std::size_t i = 0; i < sv.dim(); ++i)
    if ((i & input_mask) == 0) prob_all_zero_input += std::norm(sv.amplitude(i));
  return prob_all_zero_input > 0.5;
}

// Oracle helpers (all balanced ones here compute f(x) as an XOR of a
// fixed subset of input bits into the ancilla via CNOT -- textbook
// balanced-function constructions).
inline OracleFn constant_oracle(bool value) {
  return [value](StateVector &sv, int /*n*/, int ancilla) {
    if (value) sv.x(ancilla);
  };
}
inline OracleFn parity_oracle(int n) {
  return [n](StateVector &sv, int /*n_in*/, int ancilla) {
    for (int i = 0; i < n; ++i) sv.cnot(i, ancilla);
  };
}
inline OracleFn single_bit_oracle(int which_bit) {
  return [which_bit](StateVector &sv, int /*n*/, int ancilla) { sv.cnot(which_bit, ancilla); };
}

// ---------------------------------------------------------------------
// Grover's search: amplify the amplitude of one marked basis state among
// N = 2^n via repeated (oracle; diffusion) iterations. The oracle is
// represented directly as its required unitary action, S_f = I -
// 2|w><w| (a diagonal sign flip on the marked index) -- the standard
// "oracle as unitary" abstraction used by Grover's original algorithm
// and by textbook/library simulators alike (the O(sqrt(N))-QUERIES
// result concerns how many times this black box is invoked, not how
// many gates it costs to build from a boolean circuit).
inline void grover_oracle(StateVector &sv, std::size_t marked) { sv.set_amplitude(marked, -sv.amplitude(marked)); }

// Diffusion operator D = 2|s><s| - I, built the standard way as
// H^n (2|0><0| - I) H^n (H^n maps the |0> state to the uniform
// superposition |s> and back).
inline void grover_diffusion(StateVector &sv, int n) {
  for (int q = 0; q < n; ++q) sv.h(q);
  for (std::size_t i = 1; i < sv.dim(); ++i) sv.set_amplitude(i, -sv.amplitude(i));  // flip all but |0>
  for (int q = 0; q < n; ++q) sv.h(q);
}

inline void grover_iterate(StateVector &sv, int n, std::size_t marked) {
  grover_oracle(sv, marked);
  grover_diffusion(sv, n);
}

// Prepares the uniform superposition and runs `iterations` Grover
// iterations. Returns the resulting state (still normalized: oracle and
// diffusion are both unitary).
inline StateVector grover_search(int n, std::size_t marked, int iterations) {
  StateVector sv(n);
  for (int q = 0; q < n; ++q) sv.h(q);
  for (int it = 0; it < iterations; ++it) grover_iterate(sv, n, marked);
  return sv;
}

// The standard closed-form optimal iteration count for a single marked
// item among N = 2^n.
inline int grover_optimal_iterations(int n) {
  double big_n = static_cast<double>(std::size_t(1) << n);
  return static_cast<int>(std::floor((kPi / 4.0) * std::sqrt(big_n)));
}

// ---------------------------------------------------------------------
// Quantum Fourier Transform: |j> -> (1/sqrt(N)) sum_k e^{2*pi*i*j*k/N}
// |k>, extended by linearity to any input superposition. Textbook
// circuit: for each qubit t from most- to least-significant, apply H(t)
// then controlled phase rotations R_k = diag(1, e^{2*pi*i/2^k}) from
// every less-significant qubit c (k = t - c + 1), then reverse the qubit
// order via SWAPs (built from 3 CNOTs each, reusing the existing
// primitive rather than adding a new one) since the rotation ladder
// above produces bit-reversed output relative to our qubit-index
// convention (qubit 0 = LSB).
inline void qft_swap(StateVector &sv, int a, int b) {
  if (a == b) return;
  sv.cnot(a, b);
  sv.cnot(b, a);
  sv.cnot(a, b);
}

inline void qft(StateVector &sv, int n) {
  for (int t = n - 1; t >= 0; --t) {
    sv.h(t);
    for (int c = t - 1; c >= 0; --c) {
      int k = t - c + 1;
      double angle = 2.0 * kPi / std::pow(2.0, k);
      sv.cphase(c, t, angle);
    }
  }
  for (int i = 0; i < n / 2; ++i) qft_swap(sv, i, n - 1 - i);
}

// Brute-force O(N^2) DFT of an arbitrary amplitude vector, used purely
// as an independent ground truth to check the circuit above against.
inline std::vector<cplx> brute_force_dft(const std::vector<cplx> &input) {
  std::size_t big_n = input.size();
  std::vector<cplx> out(big_n, cplx(0.0, 0.0));
  double norm = 1.0 / std::sqrt(static_cast<double>(big_n));
  for (std::size_t k = 0; k < big_n; ++k) {
    cplx acc(0.0, 0.0);
    for (std::size_t j = 0; j < big_n; ++j) {
      double angle = 2.0 * kPi * static_cast<double>(j) * static_cast<double>(k) / static_cast<double>(big_n);
      acc += input[j] * std::polar(1.0, angle);
    }
    out[k] = acc * norm;
  }
  return out;
}

}  // namespace quantum
