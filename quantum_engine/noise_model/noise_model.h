// Phase 20 step 3: noise/decoherence channel model, applied per-gate to
// state_vector's simulator via the Monte Carlo wavefunction (quantum
// trajectory / "quantum jump") method -- Dalibard, Castin & Mollmer 1992.
// Structurally mirrors analog_engine/device_model's noise-injection
// pattern: quantum decoherence and analog device noise are different
// physics with the same "noise degrades a computed result, measure how
// much" shape.
//
// Why Monte Carlo unraveling instead of switching to a density-matrix
// representation: this simulator is a pure state vector by design (zero
// dependencies, O(2^n) memory instead of O(4^n)). Both noise channels
// below have exact Kraus decompositions with real, closed-form jump
// probabilities, so running many independent noisy trajectories and
// averaging an observable (fidelity) converges to exactly the same
// answer a full density-matrix simulation would give, at the cost of
// needing many trials instead of one -- the standard technique used by
// real quantum-trajectory simulators (e.g. QuTiP's mcsolve) for exactly
// this reason.
#pragma once

#include "../state_vector/state_vector.h"

#include <cmath>
#include <random>
#include <vector>

namespace quantum {

struct NoiseParams {
  double depolarizing_p = 0.0;          // per-gate depolarizing probability
  double amplitude_damping_gamma = 0.0;  // per-gate damping probability
};

// Depolarizing channel: rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z
// rho Z). Unraveled as: with probability (1-p) do nothing, else apply a
// uniformly random Pauli from {X, Y, Z} (each with probability p/3) --
// this is itself already a mixture of unitaries, so no renormalization
// is needed; averaging fidelity over many trajectories reproduces the
// channel's true effect exactly.
inline void apply_depolarizing(StateVector &sv, int q, double p, std::mt19937_64 &rng) {
  if (p <= 0.0) return;
  std::uniform_real_distribution<double> u(0.0, 1.0);
  if (u(rng) >= p) return;
  switch (rng() % 3) {
    case 0: sv.x(q); break;
    case 1: sv.y(q); break;
    default: sv.z(q); break;
  }
}

// Amplitude damping channel: Kraus operators K0 = diag(1, sqrt(1-gamma))
// (no-jump/"stayed" evolution) and K1 = [[0, sqrt(gamma)], [0, 0]] (jump:
// |1> decays to |0>). Jump probability is gamma * P(qubit q == 1) -- the
// standard quantum-jump result. On a jump, K1 is applied and the state
// renormalized (population collapses toward |0> on that qubit); otherwise
// K0 is applied and renormalized (the "no-jump" branch also changes the
// state, since K0 is sub-unitary -- it shrinks the |1> amplitude even
// when no jump is observed, exactly capturing "the longer you don't see
// a decay, the more likely you were already in |0>").
inline void apply_amplitude_damping(StateVector &sv, int q, double gamma, std::mt19937_64 &rng) {
  if (gamma <= 0.0) return;
  double p_jump = gamma * sv.prob1(q);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  if (p_jump > 0.0 && u(rng) < p_jump) {
    sv.apply_1q(q, cplx(0.0), cplx(std::sqrt(gamma)), cplx(0.0), cplx(0.0));
  } else {
    sv.apply_1q(q, cplx(1.0), cplx(0.0), cplx(0.0), cplx(std::sqrt(1.0 - gamma)));
  }
  sv.renormalize();
}

inline void apply_noise(StateVector &sv, int q, const NoiseParams &np, std::mt19937_64 &rng) {
  apply_depolarizing(sv, q, np.depolarizing_p, rng);
  apply_amplitude_damping(sv, q, np.amplitude_damping_gamma, rng);
}

// |<a|b>|^2 -- the standard state fidelity for two pure states.
inline double fidelity(const StateVector &a, const StateVector &b) {
  cplx acc(0.0, 0.0);
  for (std::size_t i = 0; i < a.dim(); ++i) acc += std::conj(a.amplitude(i)) * b.amplitude(i);
  return std::norm(acc);
}

// A fixed test circuit -- `depth` rounds of (H on every qubit; CNOT
// ladder qubit i -> i+1), noise applied to every qubit after every
// round when `np` is non-null. Used to measure how fidelity relative to
// the noiseless version degrades with depth and noise strength.
inline StateVector run_layered_circuit(int n, int depth, const NoiseParams *np, std::mt19937_64 *rng) {
  StateVector sv(n);
  for (int d = 0; d < depth; ++d) {
    for (int q = 0; q < n; ++q) sv.h(q);
    for (int q = 0; q + 1 < n; ++q) sv.cnot(q, q + 1);
    if (np != nullptr) {
      for (int q = 0; q < n; ++q) apply_noise(sv, q, *np, *rng);
    }
  }
  return sv;
}

inline double average_fidelity(int n, int depth, const NoiseParams &np, int trials, uint64_t seed) {
  StateVector ideal = run_layered_circuit(n, depth, nullptr, nullptr);
  double sum = 0.0;
  for (int t = 0; t < trials; ++t) {
    std::mt19937_64 rng(seed + static_cast<uint64_t>(t) * 7919);
    StateVector noisy = run_layered_circuit(n, depth, &np, &rng);
    sum += fidelity(ideal, noisy);
  }
  return sum / trials;
}

}  // namespace quantum
