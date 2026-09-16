// Phase 20 step 5: Variational Quantum Eigensolver -- the flagship hybrid
// quantum-classical algorithm. A parameterized ansatz circuit runs on
// state_vector's simulator; a classical optimizer drives its parameters
// to minimize <psi(theta)|H|psi(theta)>, the variational estimate of a
// toy Hamiltonian's ground-state energy. Checked against exact
// diagonalization (via power iteration on a dense matrix built directly
// from this simulator's own gates -- see build_pauli_string_matrix below
// -- classically tractable at the small qubit counts used here).
//
// Classical optimizer choice: hand-rolled gradient descent using the
// PARAMETER-SHIFT RULE (Mitarai, Negoro, Kitagawa & Fujii 2018) for
// exact analytic gradients, rather than importing ml/'s optimizer
// classes (LinearModel's SGD/LBFGS, BayesianOpt's GP-based search).
// Those are built around a Features/Labels supervised-learning shape;
// VQE's objective is a scalar function of a real parameter vector with
// an EXACT closed-form gradient available (no finite-difference
// approximation needed, unlike sciml/neural_ode's or ssm_layer's
// finite-difference GD), so a small hand-rolled loop using that exact
// gradient is both simpler and more accurate than adapting either ml/
// class to a problem shape neither was built for.
#pragma once

#include "../state_vector/state_vector.h"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace quantum {

// Pauli operator tags for building a Hamiltonian as a sum of Pauli
// strings (the standard qubit-Hamiltonian representation after e.g.
// Jordan-Wigner mapping).
enum class Pauli { I, X, Y, Z };

struct PauliTerm {
  double coeff;
  std::vector<Pauli> ops;  // one entry per qubit
};

using DenseMatrix = std::vector<std::vector<cplx>>;

// Builds the dense 2^n x 2^n matrix for ONE Pauli string by probing this
// simulator's OWN gates on every computational basis vector (column j is
// the image of |j>) -- guarantees the matrix is exactly consistent with
// how state_vector.h's x()/y()/z() actually act, with no separate
// Kronecker-product bookkeeping (and no risk of a qubit-ordering
// mismatch between "how I derived the matrix by hand" and "how the
// simulator actually indexes qubits") to get wrong.
inline DenseMatrix build_pauli_string_matrix(const std::vector<Pauli> &ops, int n) {
  std::size_t dim = std::size_t(1) << n;
  DenseMatrix m(dim, std::vector<cplx>(dim, cplx(0.0, 0.0)));
  for (std::size_t basis = 0; basis < dim; ++basis) {
    StateVector sv(n);
    sv.set_amplitude(0, cplx(0.0, 0.0));
    sv.set_amplitude(basis, cplx(1.0, 0.0));
    for (int q = 0; q < n; ++q) {
      switch (ops[static_cast<std::size_t>(q)]) {
        case Pauli::I: break;
        case Pauli::X: sv.x(q); break;
        case Pauli::Y: sv.y(q); break;
        case Pauli::Z: sv.z(q); break;
      }
    }
    for (std::size_t row = 0; row < dim; ++row) m[row][basis] = sv.amplitude(row);
  }
  return m;
}

inline DenseMatrix hamiltonian_dense(const std::vector<PauliTerm> &terms, int n) {
  std::size_t dim = std::size_t(1) << n;
  DenseMatrix h(dim, std::vector<cplx>(dim, cplx(0.0, 0.0)));
  for (const auto &term : terms) {
    DenseMatrix pm = build_pauli_string_matrix(term.ops, n);
    for (std::size_t i = 0; i < dim; ++i)
      for (std::size_t j = 0; j < dim; ++j) h[i][j] += term.coeff * pm[i][j];
  }
  return h;
}

inline std::vector<cplx> mat_vec(const DenseMatrix &m, const std::vector<cplx> &v) {
  std::vector<cplx> out(v.size(), cplx(0.0, 0.0));
  for (std::size_t i = 0; i < m.size(); ++i) {
    cplx acc(0.0, 0.0);
    for (std::size_t j = 0; j < v.size(); ++j) acc += m[i][j] * v[j];
    out[i] = acc;
  }
  return out;
}

inline cplx inner(const std::vector<cplx> &a, const std::vector<cplx> &b) {
  cplx acc(0.0, 0.0);
  for (std::size_t i = 0; i < a.size(); ++i) acc += std::conj(a[i]) * b[i];
  return acc;
}

// Exact ground-state energy via power iteration on (shift*I - H): the
// TOP eigenvalue of the shifted operator corresponds to H's SMALLEST
// eigenvalue, energy = shift - top_eigenvalue. `shift` must exceed H's
// largest-magnitude eigenvalue; sum(|coeff|) + 1.0 is a valid bound
// since every Pauli string has operator norm exactly 1.
inline double exact_ground_energy(const std::vector<PauliTerm> &terms, int n, int iterations = 2000) {
  DenseMatrix h = hamiltonian_dense(terms, n);
  std::size_t dim = h.size();
  double shift = 1.0;
  for (const auto &t : terms) shift += std::abs(t.coeff);

  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<cplx> v(dim);
  for (auto &x : v) x = cplx(u(rng), 0.0);
  // NOTE: inner(v, v) is already the real scalar sum|v_i|^2 (the squared
  // vector norm) -- taking std::norm() of it (which squares a COMPLEX
  // scalar's modulus) would square it again. Use .real() directly.
  double norm = std::sqrt(inner(v, v).real());
  for (auto &x : v) x /= norm;

  double eigenvalue = 0.0;
  for (int it = 0; it < iterations; ++it) {
    std::vector<cplx> hv = mat_vec(h, v);
    std::vector<cplx> shifted(dim);
    for (std::size_t i = 0; i < dim; ++i) shifted[i] = shift * v[i] - hv[i];
    double nrm = std::sqrt(inner(shifted, shifted).real());
    for (auto &x : shifted) x /= nrm;
    v = shifted;
    eigenvalue = nrm;  // Rayleigh-quotient-equivalent for a converged unit eigenvector under power iteration
  }
  return shift - eigenvalue;
}

// Hardware-efficient ansatz: `depth` layers of (RY on every qubit; CNOT
// ladder). Parameter count is n * depth.
inline StateVector vqe_ansatz(const std::vector<double> &theta, int n, int depth) {
  StateVector sv(n);
  std::size_t idx = 0;
  for (int d = 0; d < depth; ++d) {
    for (int q = 0; q < n; ++q) sv.ry(q, theta[idx++]);
    for (int q = 0; q + 1 < n; ++q) sv.cnot(q, q + 1);
  }
  return sv;
}

inline double vqe_expectation(const std::vector<double> &theta, int n, int depth, const DenseMatrix &h) {
  StateVector sv = vqe_ansatz(theta, n, depth);
  std::vector<cplx> hpsi = mat_vec(h, sv.amplitudes());
  return inner(sv.amplitudes(), hpsi).real();
}

// Exact analytic gradient via the parameter-shift rule (Mitarai et al.
// 2018): for a gate generated as exp(-i*theta*G/2) with G having
// eigenvalues +-1 (true of RY = exp(-i*theta*Y/2), state_vector.h's
// convention exactly), d<H>/dtheta_i = 0.5*(E(theta_i + pi/2) -
// E(theta_i - pi/2)) EXACTLY, no finite-difference approximation.
inline std::vector<double> vqe_gradient(const std::vector<double> &theta, int n, int depth, const DenseMatrix &h) {
  std::vector<double> grad(theta.size(), 0.0);
  for (std::size_t i = 0; i < theta.size(); ++i) {
    std::vector<double> plus = theta, minus = theta;
    plus[i] += kPi / 2.0;
    minus[i] -= kPi / 2.0;
    grad[i] = 0.5 * (vqe_expectation(plus, n, depth, h) - vqe_expectation(minus, n, depth, h));
  }
  return grad;
}

struct VqeResult {
  std::vector<double> theta;
  std::vector<double> energy_trace;
};

inline VqeResult vqe_optimize(int n, int depth, const DenseMatrix &h, int iterations, double lr, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-0.1, 0.1);
  std::vector<double> theta(static_cast<std::size_t>(n * depth));
  for (auto &t : theta) t = u(rng);

  VqeResult result;
  for (int it = 0; it < iterations; ++it) {
    double e = vqe_expectation(theta, n, depth, h);
    result.energy_trace.push_back(e);
    std::vector<double> grad = vqe_gradient(theta, n, depth, h);
    for (std::size_t i = 0; i < theta.size(); ++i) theta[i] -= lr * grad[i];
  }
  result.theta = theta;
  return result;
}

}  // namespace quantum
