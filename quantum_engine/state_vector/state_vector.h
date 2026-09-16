// Phase 20 step 1: hand-rolled state-vector quantum simulator. Zero
// dependencies -- a real 2^n-amplitude complex state vector, single-qubit
// gates applied as structured 2x2 tensor contractions over amplitude pairs,
// two-qubit controlled gates applied as index-masked selection, and
// measurement/sampling. Same "hand-roll the primitive first" precedent as
// foundation/proptest and observability/opentelemetry.
//
// Qubit indexing: qubit 0 is the least-significant bit of the basis-state
// index, i.e. basis state |q_{n-1} ... q_1 q_0> maps to index
// sum_i q_i * 2^i. This is the opposite convention from Nielsen & Chuang's
// usual |q_0 q_1 ... q_{n-1}> left-to-right notation, chosen because it
// makes the LSB-first CMakeLists-free bit trick below (index & bit) direct.
#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace quantum {

using cplx = std::complex<double>;
inline constexpr double kPi = 3.14159265358979323846;

class StateVector {
 public:
  explicit StateVector(int n_qubits) : n_(n_qubits), amps_(std::size_t(1) << n_qubits, cplx(0.0, 0.0)) {
    if (n_qubits < 1 || n_qubits > 24) throw std::invalid_argument("StateVector: n_qubits must be in [1, 24]");
    amps_[0] = cplx(1.0, 0.0);  // |0...0>
  }

  int num_qubits() const { return n_; }
  std::size_t dim() const { return amps_.size(); }
  const std::vector<cplx> &amplitudes() const { return amps_; }
  cplx amplitude(std::size_t basis_index) const { return amps_.at(basis_index); }
  void set_amplitude(std::size_t basis_index, cplx v) { amps_.at(basis_index) = v; }

  double norm_squared() const {
    double s = 0.0;
    for (const auto &a : amps_) s += std::norm(a);
    return s;
  }

  // Divides every amplitude by sqrt(norm_squared()). Needed after
  // applying a sub-unitary (non-norm-preserving) operator, e.g. one Kraus
  // operator of a noise channel's Monte Carlo wavefunction unraveling
  // (see noise_model.h) -- a genuine physical renormalization step, not a
  // numerical-stability nicety.
  void renormalize() {
    double n2 = norm_squared();
    if (n2 <= 0.0) throw std::runtime_error("StateVector::renormalize: zero-norm state");
    double inv = 1.0 / std::sqrt(n2);
    for (auto &a : amps_) a *= inv;
  }

  // Applies a general single-qubit unitary [[m00,m01],[m10,m11]] to `qubit`.
  // Iterates only over the half of the amplitude vector with bit `qubit`
  // clear, updating each (i, i|bit) pair together -- O(2^n) per gate, no
  // allocation.
  void apply_1q(int qubit, cplx m00, cplx m01, cplx m10, cplx m11) {
    std::size_t bit = std::size_t(1) << qubit;
    for (std::size_t i = 0; i < amps_.size(); ++i) {
      if (i & bit) continue;
      std::size_t j = i | bit;
      cplx a0 = amps_[i], a1 = amps_[j];
      amps_[i] = m00 * a0 + m01 * a1;
      amps_[j] = m10 * a0 + m11 * a1;
    }
  }

  void x(int q) { apply_1q(q, cplx(0), cplx(1), cplx(1), cplx(0)); }
  void y(int q) { apply_1q(q, cplx(0), cplx(0, -1), cplx(0, 1), cplx(0)); }
  void z(int q) { apply_1q(q, cplx(1), cplx(0), cplx(0), cplx(-1)); }
  void h(int q) {
    double s = 1.0 / std::sqrt(2.0);
    apply_1q(q, cplx(s), cplx(s), cplx(s), cplx(-s));
  }
  void rx(int q, double theta) {
    double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
    apply_1q(q, cplx(c, 0), cplx(0, -s), cplx(0, -s), cplx(c, 0));
  }
  void ry(int q, double theta) {
    double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
    apply_1q(q, cplx(c, 0), cplx(-s, 0), cplx(s, 0), cplx(c, 0));
  }
  void rz(int q, double theta) {
    apply_1q(q, std::polar(1.0, -theta / 2.0), cplx(0), cplx(0), std::polar(1.0, theta / 2.0));
  }
  // diag(1, e^{i theta}) -- the phase gate QFT's controlled version reuses.
  void phase(int q, double theta) { apply_1q(q, cplx(1), cplx(0), cplx(0), std::polar(1.0, theta)); }

  void cnot(int control, int target) {
    std::size_t cbit = std::size_t(1) << control, tbit = std::size_t(1) << target;
    for (std::size_t i = 0; i < amps_.size(); ++i) {
      if (!(i & cbit) || (i & tbit)) continue;
      std::swap(amps_[i], amps_[i | tbit]);
    }
  }

  void cz(int control, int target) {
    std::size_t cbit = std::size_t(1) << control, tbit = std::size_t(1) << target;
    for (std::size_t i = 0; i < amps_.size(); ++i)
      if ((i & cbit) && (i & tbit)) amps_[i] = -amps_[i];
  }

  void cphase(int control, int target, double theta) {
    std::size_t cbit = std::size_t(1) << control, tbit = std::size_t(1) << target;
    cplx factor = std::polar(1.0, theta);
    for (std::size_t i = 0; i < amps_.size(); ++i)
      if ((i & cbit) && (i & tbit)) amps_[i] *= factor;
  }

  // Controlled-controlled-X (Toffoli), used by QEC's syndrome-based
  // correction step. control2 is the "more significant" control in no
  // particular sense -- both controls are symmetric.
  void ccx(int control1, int control2, int target) {
    std::size_t c1 = std::size_t(1) << control1, c2 = std::size_t(1) << control2, tbit = std::size_t(1) << target;
    for (std::size_t i = 0; i < amps_.size(); ++i) {
      if (!(i & c1) || !(i & c2) || (i & tbit)) continue;
      std::swap(amps_[i], amps_[i | tbit]);
    }
  }

  double prob1(int q) const {
    std::size_t bit = std::size_t(1) << q;
    double p = 0.0;
    for (std::size_t i = 0; i < amps_.size(); ++i)
      if (i & bit) p += std::norm(amps_[i]);
    return p;
  }

  // Zeros every amplitude inconsistent with qubit `q` measuring `outcome`
  // (0 or 1), WITHOUT renormalizing. Used to compute exact conditional
  // amplitude ratios (renormalization-invariant) for deterministic circuit-
  // identity tests, e.g. teleportation. Real measurement (measure_all)
  // renormalizes; this is the same physics, just left unnormalized so
  // ratios stay exact instead of accumulating a sqrt rounding step.
  void project_unnormalized(int q, int outcome) {
    std::size_t bit = std::size_t(1) << q;
    for (std::size_t i = 0; i < amps_.size(); ++i) {
      int bit_val = (i & bit) ? 1 : 0;
      if (bit_val != outcome) amps_[i] = cplx(0.0, 0.0);
    }
  }

  // Samples a full computational-basis outcome from |amp|^2 and collapses
  // the state vector to that basis state (renormalized to a pure |1>
  // amplitude, matching the Born rule's post-measurement state).
  std::size_t measure_all(std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng), cum = 0.0;
    std::size_t outcome = amps_.size() - 1;
    for (std::size_t i = 0; i < amps_.size(); ++i) {
      cum += std::norm(amps_[i]);
      if (r <= cum) { outcome = i; break; }
    }
    for (std::size_t i = 0; i < amps_.size(); ++i) amps_[i] = (i == outcome) ? cplx(1.0, 0.0) : cplx(0.0, 0.0);
    return outcome;
  }

 private:
  int n_;
  std::vector<cplx> amps_;
};

}  // namespace quantum
