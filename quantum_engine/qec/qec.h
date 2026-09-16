// Phase 20 step 4: minimal quantum error correction -- the 3-qubit
// bit-flip repetition code, demonstrated actually protecting a logical
// qubit against injected noise, with logical vs. physical error rate
// measured directly (not just argued).
//
// Register layout (5 qubits): q0 = logical data qubit (also the
// surviving data qubit after decoding), q1/q2 = the two redundant
// physical qubits added by encoding, q3/q4 = syndrome-measurement
// ancillas.
//
// Why this doesn't need ancilla-free/ideal-measurement shortcuts: after
// encoding, the state is always of the form a|e0 e1 e2> + b|~e0 ~e1 ~e2>
// for some error bit pattern (e0,e1,e2) -- independent X (bit-flip)
// errors are unitary permutations of computational basis states, so they
// never create additional superposition beyond the original a/b logical
// amplitudes. The pairwise parities e0^e1 and e1^e2 are therefore
// IDENTICAL for the complemented pattern (~e0^~e1 = e0^e1), so measuring
// them via ancilla CNOTs (the real stabilizer-measurement circuit for
// Z0Z1 and Z1Z2) reveals the error location without ever collapsing the
// a/b logical superposition -- the actual mechanism that makes QEC work,
// not a shortcut around it.
#pragma once

#include "../noise_model/noise_model.h"
#include "../state_vector/state_vector.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <utility>

namespace quantum {

inline void qec_encode(StateVector &sv) {
  sv.cnot(0, 1);
  sv.cnot(0, 2);
}

inline void qec_decode(StateVector &sv) {
  sv.cnot(0, 2);
  sv.cnot(0, 1);
}

// Independent bit-flip (X) error on each of the 3 data qubits with
// probability p_phys each -- the textbook noise model the repetition
// code is designed against.
inline void qec_inject_bitflip_errors(StateVector &sv, double p_phys, std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int q = 0; q < 3; ++q)
    if (u(rng) < p_phys) sv.x(q);
}

// A harsher, more realistic noise model: full depolarizing noise (X, Y,
// AND Z errors, not just X) on each data qubit, reusing noise_model.h's
// channel unmodified. The bit-flip code only targets X errors, so this
// is expected to protect much less well -- see qec_test.cpp's second
// test and the README's disclosed finding.
inline void qec_inject_depolarizing_errors(StateVector &sv, double p_phys, std::mt19937_64 &rng) {
  for (int q = 0; q < 3; ++q) apply_depolarizing(sv, q, p_phys, rng);
}

// The real stabilizer-measurement circuit: CNOTs from data qubits into
// syndrome ancillas compute q3 = q0^q1, q4 = q1^q2 without an
// intervening Hadamard, so the ancillas end up in a definite
// computational-basis value correlated with the error pattern -- no
// superposition is created on the ancillas themselves.
inline void qec_extract_syndrome(StateVector &sv) {
  sv.cnot(0, 3);
  sv.cnot(1, 3);
  sv.cnot(1, 4);
  sv.cnot(2, 4);
}

inline std::pair<int, int> qec_measure_syndrome(StateVector &sv, std::mt19937_64 &rng) {
  int s1 = sv.measure_qubit(3, rng);
  int s2 = sv.measure_qubit(4, rng);
  return {s1, s2};
}

// Standard 3-qubit bit-flip code syndrome table for stabilizers Z0Z1,
// Z1Z2: (0,0)=no error, (1,0)=qubit0, (1,1)=qubit1, (0,1)=qubit2.
inline void qec_correct(StateVector &sv, int s1, int s2) {
  if (s1 == 1 && s2 == 0) sv.x(0);
  else if (s1 == 1 && s2 == 1) sv.x(1);
  else if (s1 == 0 && s2 == 1) sv.x(2);
  // (0,0): no error, no correction.
}

// Runs one full encode -> inject error -> syndrome -> correct -> decode
// trial for a logical qubit prepared via ry(theta), and returns whether
// the recovered qubit matches the original exactly (up to the syndrome
// ancillas' own measured values, which are classical bits unrelated to
// the logical amplitude and are accounted for in the index check).
// `inject` is the caller-supplied error-injection function, so the same
// harness drives both the bit-flip-only and general-depolarizing
// experiments below.
using ErrorInjector = std::function<void(StateVector &, double, std::mt19937_64 &)>;

inline bool qec_trial_succeeds(double theta, double p_phys, const ErrorInjector &inject, std::mt19937_64 &rng) {
  StateVector sv(5);
  sv.ry(0, theta);
  qec_encode(sv);
  inject(sv, p_phys, rng);
  qec_extract_syndrome(sv);
  auto [s1, s2] = qec_measure_syndrome(sv, rng);
  qec_correct(sv, s1, s2);
  qec_decode(sv);

  std::size_t idx0 = (static_cast<std::size_t>(s1) << 3) | (static_cast<std::size_t>(s2) << 4);
  std::size_t idx1 = idx0 | 1;
  double mass_at_expected = std::norm(sv.amplitude(idx0)) + std::norm(sv.amplitude(idx1));
  if (mass_at_expected < 0.999) return false;  // logical error: probability leaked elsewhere

  double expected_ratio = std::tan(theta / 2.0);
  double actual_ratio = sv.amplitude(idx0).real() != 0.0 ? sv.amplitude(idx1).real() / sv.amplitude(idx0).real()
                                                          : std::numeric_limits<double>::infinity();
  return std::abs(actual_ratio - expected_ratio) < 1e-6;
}

// Estimates the logical error rate over `trials` independent runs.
inline double qec_logical_error_rate(double theta, double p_phys, const ErrorInjector &inject, int trials, uint64_t seed) {
  int failures = 0;
  for (int t = 0; t < trials; ++t) {
    std::mt19937_64 rng(seed + static_cast<uint64_t>(t) * 104729);
    if (!qec_trial_succeeds(theta, p_phys, inject, rng)) ++failures;
  }
  return static_cast<double>(failures) / trials;
}

}  // namespace quantum
