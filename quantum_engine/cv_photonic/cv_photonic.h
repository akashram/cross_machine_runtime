// Phase 20 step 8: continuous-variable (CV) / photonic quantum computing
// primitives -- Xanadu's specific hardware approach, genuinely different
// from the qubit-based model in steps 1-7 (qumodes instead of qubits,
// Gaussian states instead of a finite-dimensional state vector, a
// different native gate set). Hand-rolled via Gaussian-state
// covariance-matrix linear algebra (Weedbrook, Pirandola, Garcia-Patron,
// Cerf, Ralph, Shapiro & Lloyd 2012, "Gaussian Quantum Information",
// Rev. Mod. Phys. 84) -- no new dependency needed, per PLAN.md step 8's
// explicit preference for the hand-rolled path over Strawberry Fields
// (the user declined installing Strawberry Fields this session; see
// README for the toolchain-gated alternative note).
//
// Convention: quadratures ordered (x_1, p_1, x_2, p_2, ..., x_n, p_n),
// [x, p] = i (hbar = 1), vacuum covariance = I (each mode's vacuum
// variance is 1 in both quadratures) -- a standard, self-consistent
// choice; picked explicitly here rather than assumed to match any
// specific library's default hbar convention (Strawberry Fields
// defaults to hbar=2, which would rescale variances by a constant
// factor -- irrelevant to every check in this step, which all use
// RATIOS or closed-form combinations that are convention-independent
// where it matters, and are explicit about which convention is in use
// where it does).
#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace quantum::cv {

using Vec = std::vector<double>;
using Mat = std::vector<std::vector<double>>;

inline Mat identity(int d) {
  Mat m(static_cast<std::size_t>(d), Vec(static_cast<std::size_t>(d), 0.0));
  for (int i = 0; i < d; ++i) m[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
  return m;
}

inline Mat matmul(const Mat &a, const Mat &b) {
  std::size_t n = a.size(), k = b.size(), m = b[0].size();
  Mat out(n, Vec(m, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t p = 0; p < k; ++p) {
      double aip = a[i][p];
      for (std::size_t j = 0; j < m; ++j) out[i][j] += aip * b[p][j];
    }
  return out;
}

inline Mat transpose(const Mat &a) {
  std::size_t n = a.size(), m = a[0].size();
  Mat out(m, Vec(n, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < m; ++j) out[j][i] = a[i][j];
  return out;
}

inline Vec matvec(const Mat &a, const Vec &v) {
  Vec out(a.size(), 0.0);
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < v.size(); ++j) out[i] += a[i][j] * v[j];
  return out;
}

// The symplectic form for quadrature ordering (x_1,p_1,...,x_n,p_n):
// block-diagonal [[0,1],[-1,0]] per mode.
inline Mat symplectic_form(int n_modes) {
  int d = 2 * n_modes;
  Mat omega(static_cast<std::size_t>(d), Vec(static_cast<std::size_t>(d), 0.0));
  for (int m = 0; m < n_modes; ++m) {
    omega[static_cast<std::size_t>(2 * m)][static_cast<std::size_t>(2 * m + 1)] = 1.0;
    omega[static_cast<std::size_t>(2 * m + 1)][static_cast<std::size_t>(2 * m)] = -1.0;
  }
  return omega;
}

// A matrix S is symplectic iff S*Omega*S^T = Omega -- the defining
// property that guarantees a Gaussian transformation preserves the
// canonical commutation relations (and hence maps physical states to
// physical states).
inline bool is_symplectic(const Mat &s, double tol = 1e-9) {
  int d = static_cast<int>(s.size());
  Mat omega = symplectic_form(d / 2);
  Mat lhs = matmul(matmul(s, omega), transpose(s));
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j)
      if (std::abs(lhs[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] -
                    omega[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]) > tol)
        return false;
  return true;
}

struct GaussianState {
  int n_modes;
  Vec mean;  // size 2*n_modes
  Mat cov;   // size 2*n_modes x 2*n_modes
};

inline GaussianState vacuum_state(int n_modes) {
  GaussianState s;
  s.n_modes = n_modes;
  s.mean.assign(static_cast<std::size_t>(2 * n_modes), 0.0);
  s.cov = identity(2 * n_modes);
  return s;
}

// Embeds a local symplectic matrix (acting on the quadrature indices
// listed in `quad_indices`) into the full 2n-dimensional identity, then
// applies it: cov -> S cov S^T, mean -> S mean.
inline void apply_symplectic(GaussianState &s, const Mat &local, const std::vector<int> &quad_indices) {
  int d = 2 * s.n_modes;
  Mat full = identity(d);
  for (std::size_t a = 0; a < quad_indices.size(); ++a)
    for (std::size_t b = 0; b < quad_indices.size(); ++b)
      full[static_cast<std::size_t>(quad_indices[a])][static_cast<std::size_t>(quad_indices[b])] = local[a][b];
  s.cov = matmul(matmul(full, s.cov), transpose(full));
  s.mean = matvec(full, s.mean);
}

// Displacement: a pure translation of the mean vector, covariance
// unchanged -- not a linear (matrix) symplectic transformation, applied
// directly.
inline void displace(GaussianState &s, int mode, double dx, double dp) {
  s.mean[static_cast<std::size_t>(2 * mode)] += dx;
  s.mean[static_cast<std::size_t>(2 * mode + 1)] += dp;
}

// Single-mode squeezing: S_sq(r) = diag(e^-r, e^r), squeezes the x
// quadrature's variance by e^-2r and stretches p's by e^2r -- an ideal
// (noiseless) squeezer stays a minimum-uncertainty state, Vx*Vp = 1
// exactly, checked directly in cv_photonic_test.cpp.
inline Mat squeeze_local_matrix(double r) { return {{std::exp(-r), 0.0}, {0.0, std::exp(r)}}; }

inline void squeeze(GaussianState &s, int mode, double r) {
  apply_symplectic(s, squeeze_local_matrix(r), {2 * mode, 2 * mode + 1});
}

// A (phaseless) 50:50-family beamsplitter mixing modes i and j by angle
// theta, acting on (x_i, p_i, x_j, p_j) as a rotation-like symplectic
// block. A real, passive (photon-number-conserving) linear-optical
// element -- checked directly against that conservation law in
// cv_photonic_test.cpp.
inline Mat beamsplitter_local_matrix(double theta) {
  double c = std::cos(theta), sn = std::sin(theta);
  return {{c, 0.0, sn, 0.0}, {0.0, c, 0.0, sn}, {-sn, 0.0, c, 0.0}, {0.0, -sn, 0.0, c}};
}

inline void beamsplitter(GaussianState &s, int mode_i, int mode_j, double theta) {
  apply_symplectic(s, beamsplitter_local_matrix(theta), {2 * mode_i, 2 * mode_i + 1, 2 * mode_j, 2 * mode_j + 1});
}

// Mean photon number of one mode: n_bar = (Vx+Vp)/4 - 1/2 + (dx^2+dp^2)/2
// -- the standard result for a Gaussian state with vacuum variance 1
// (checked in the test against the closed-form sinh^2(r) result for
// squeezed vacuum specifically).
inline double mean_photon_number(const GaussianState &s, int mode) {
  double vx = s.cov[static_cast<std::size_t>(2 * mode)][static_cast<std::size_t>(2 * mode)];
  double vp = s.cov[static_cast<std::size_t>(2 * mode + 1)][static_cast<std::size_t>(2 * mode + 1)];
  double dx = s.mean[static_cast<std::size_t>(2 * mode)];
  double dp = s.mean[static_cast<std::size_t>(2 * mode + 1)];
  return (vx + vp) / 4.0 - 0.5 + (dx * dx + dp * dp) / 2.0;
}

inline double total_photon_number(const GaussianState &s) {
  double total = 0.0;
  for (int m = 0; m < s.n_modes; ++m) total += mean_photon_number(s, m);
  return total;
}

// Homodyne measurement of one quadrature of one mode: samples from that
// quadrature's MARGINAL Gaussian (mean, variance read directly off the
// state). Disclosed simplification: this does not implement the full
// Gaussian conditional-state collapse of the OTHER modes after
// measurement (the standard Schur-complement formula) -- only the
// measured quadrature's own marginal statistics, which is what's needed
// to demonstrate the measurement primitive itself. See README.
inline double homodyne_sample(const GaussianState &s, int mode, char quadrature, std::mt19937_64 &rng) {
  int idx = (quadrature == 'x') ? 2 * mode : 2 * mode + 1;
  double mean = s.mean[static_cast<std::size_t>(idx)];
  double var = s.cov[static_cast<std::size_t>(idx)][static_cast<std::size_t>(idx)];
  std::normal_distribution<double> dist(mean, std::sqrt(var));
  return dist(rng);
}

}  // namespace quantum::cv
