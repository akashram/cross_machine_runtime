//===- ssm_layer.h - linear state-space layer vs. self-attention --------===//
//
// PLAN.md Phase 18 step 6: Gu, Goel & Ré (2021), "Efficiently Modeling
// Long Sequences with Structured State Spaces" (S4) -- a real linear
// state-space recurrence layer, benchmarked directly against a
// self-attention layer on a small sequence task, for both accuracy and
// MEASURED (instrumented multiply-add counting, not a cited Big-O
// formula) asymptotic compute: O(L) for the SSM's recurrence vs. O(L^2)
// for attention's pairwise score matrix.
//
// Scope note: this step implements a standard scaled dot-product
// self-attention layer directly (Vaswani et al. 2017's formula, single
// head, over plain std::vector<double> sequences) rather than gluing
// together `distributed_training/tensor_parallel_attn`'s Tensor/autograd-
// based module, which is built for that phase's multi-rank tensor-
// parallel TRAINING pipeline, not a quick op-count/accuracy comparison
// against a from-scratch SSM operating on this phase's plain State
// vectors. Both layers here are trained the SAME way (finite-difference
// gradient descent over a flattened parameter vector) specifically so the
// comparison isn't confounded by one model getting a hand-derived exact
// backward pass and the other an approximate one -- a real, disclosed
// scope decision, not an oversight.
//
// The SSM here is a GENERIC (randomly initialized) linear recurrence, not
// S4's HiPPO-structured state matrix -- see ssm_layer_test.cpp and
// README.md for what that choice does and doesn't let the model learn on
// the long-range copy task below.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cmath>
#include <random>
#include <vector>

namespace sciml {

using Vec = std::vector<double>;
using Seq = std::vector<Vec>;

// Instrumented multiply-add counter -- reset before a forward pass, read
// after, for a REAL measured operation count rather than a cited formula.
// Single-threaded use only (this repo's test binaries are single-threaded
// throughout).
inline long &op_counter() {
  static long counter = 0;
  return counter;
}
inline void reset_op_counter() { op_counter() = 0; }

// ---------------------------------------------------------------------
// Linear state-space layer: x_{t+1} = A*x_t + B*u_t, y_t = C*x_t.
// State dim n, input/output dim d. O(L) forward pass: one recurrence step
// per timestep, each O(n^2 + n*d + n*d) work, independent of L elsewhere.
// ---------------------------------------------------------------------
struct SSMParams {
  int n = 4, d = 3; // state dim, input/output dim
  std::vector<double> A, B, C; // sizes n*n, n*d, d*n

  int num_params() const { return n * n + n * d + d * n; }

  static SSMParams random_init(int n, int d, uint64_t seed) {
    SSMParams p;
    p.n = n;
    p.d = d;
    std::mt19937_64 rng(seed);
    // A scaled down so the recurrence doesn't blow up over long sequences
    // -- a GENERIC stability choice, not S4's HiPPO structure (see file
    // header).
    std::normal_distribution<double> da(0.0, 0.5 / std::sqrt(static_cast<double>(n)));
    std::normal_distribution<double> dbc(0.0, 0.5);
    p.A.resize(static_cast<size_t>(n * n));
    p.B.resize(static_cast<size_t>(n * d));
    p.C.resize(static_cast<size_t>(d * n));
    for (double &v : p.A) v = da(rng);
    for (double &v : p.B) v = dbc(rng);
    for (double &v : p.C) v = dbc(rng);
    return p;
  }

  std::vector<double> flatten() const {
    std::vector<double> out;
    out.insert(out.end(), A.begin(), A.end());
    out.insert(out.end(), B.begin(), B.end());
    out.insert(out.end(), C.begin(), C.end());
    return out;
  }
  static SSMParams from_flat(const std::vector<double> &flat, int n, int d) {
    SSMParams p;
    p.n = n;
    p.d = d;
    size_t off = 0;
    p.A.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n * n)));
    off += static_cast<size_t>(n * n);
    p.B.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(n * d)));
    off += static_cast<size_t>(n * d);
    p.C.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + static_cast<size_t>(d * n)));
    return p;
  }
};

inline Seq ssm_forward(const SSMParams &p, const Seq &u) {
  int n = p.n, d = p.d;
  std::vector<double> x(static_cast<size_t>(n), 0.0);
  Seq y;
  y.reserve(u.size());
  for (const Vec &ut : u) {
    std::vector<double> x_next(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
      double s = 0.0;
      for (int j = 0; j < n; ++j) {
        s += p.A[static_cast<size_t>(i * n + j)] * x[static_cast<size_t>(j)];
        ++op_counter();
      }
      for (int j = 0; j < d; ++j) {
        s += p.B[static_cast<size_t>(i * d + j)] * ut[static_cast<size_t>(j)];
        ++op_counter();
      }
      x_next[static_cast<size_t>(i)] = s;
    }
    x = x_next;
    Vec yt(static_cast<size_t>(d), 0.0);
    for (int i = 0; i < d; ++i) {
      double s = 0.0;
      for (int j = 0; j < n; ++j) {
        s += p.C[static_cast<size_t>(i * n + j)] * x[static_cast<size_t>(j)];
        ++op_counter();
      }
      yt[static_cast<size_t>(i)] = s;
    }
    y.push_back(yt);
  }
  return y;
}

// ---------------------------------------------------------------------
// Single-head scaled dot-product self-attention (Vaswani et al. 2017).
// O(L^2) forward pass: the L x L score matrix is unavoidable.
// ---------------------------------------------------------------------
struct AttnParams {
  int d = 3;
  std::vector<double> Wq, Wk, Wv, Wo; // each d*d

  int num_params() const { return 4 * d * d; }

  static AttnParams random_init(int d, uint64_t seed) {
    AttnParams p;
    p.d = d;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.5 / std::sqrt(static_cast<double>(d)));
    p.Wq.resize(static_cast<size_t>(d * d));
    p.Wk.resize(static_cast<size_t>(d * d));
    p.Wv.resize(static_cast<size_t>(d * d));
    p.Wo.resize(static_cast<size_t>(d * d));
    for (auto *m : {&p.Wq, &p.Wk, &p.Wv, &p.Wo})
      for (double &v : *m) v = dist(rng);
    return p;
  }

  std::vector<double> flatten() const {
    std::vector<double> out;
    out.insert(out.end(), Wq.begin(), Wq.end());
    out.insert(out.end(), Wk.begin(), Wk.end());
    out.insert(out.end(), Wv.begin(), Wv.end());
    out.insert(out.end(), Wo.begin(), Wo.end());
    return out;
  }
  static AttnParams from_flat(const std::vector<double> &flat, int d) {
    AttnParams p;
    p.d = d;
    size_t sz = static_cast<size_t>(d * d), off = 0;
    p.Wq.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + sz));
    off += sz;
    p.Wk.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + sz));
    off += sz;
    p.Wv.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + sz));
    off += sz;
    p.Wo.assign(flat.begin() + static_cast<long>(off), flat.begin() + static_cast<long>(off + sz));
    return p;
  }
};

inline Vec matvec(const std::vector<double> &W, int d, const Vec &x) {
  Vec out(static_cast<size_t>(d), 0.0);
  for (int i = 0; i < d; ++i) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) {
      s += W[static_cast<size_t>(i * d + j)] * x[static_cast<size_t>(j)];
      ++op_counter();
    }
    out[static_cast<size_t>(i)] = s;
  }
  return out;
}

inline Seq attn_forward(const AttnParams &p, const Seq &u) {
  int d = p.d;
  size_t L = u.size();
  std::vector<Vec> Q(L), K(L), V(L);
  for (size_t t = 0; t < L; ++t) {
    Q[t] = matvec(p.Wq, d, u[t]);
    K[t] = matvec(p.Wk, d, u[t]);
    V[t] = matvec(p.Wv, d, u[t]);
  }
  double scale = 1.0 / std::sqrt(static_cast<double>(d));
  Seq y(L);
  for (size_t i = 0; i < L; ++i) {
    std::vector<double> scores(L, 0.0);
    double max_score = -1e300;
    for (size_t j = 0; j < L; ++j) {
      double s = 0.0;
      for (int k = 0; k < d; ++k) {
        s += Q[i][static_cast<size_t>(k)] * K[j][static_cast<size_t>(k)];
        ++op_counter();
      }
      scores[j] = s * scale;
      max_score = std::max(max_score, scores[j]);
    }
    double denom = 0.0;
    for (size_t j = 0; j < L; ++j) {
      scores[j] = std::exp(scores[j] - max_score);
      denom += scores[j];
    }
    Vec context(static_cast<size_t>(d), 0.0);
    for (size_t j = 0; j < L; ++j) {
      double w = scores[j] / denom;
      for (int k = 0; k < d; ++k) {
        context[static_cast<size_t>(k)] += w * V[j][static_cast<size_t>(k)];
        ++op_counter();
      }
    }
    y[i] = matvec(p.Wo, d, context);
  }
  return y;
}

// ---------------------------------------------------------------------
// Generic finite-difference gradient descent over a flattened parameter
// vector -- used identically for both models so the comparison isn't
// confounded by one getting a hand-derived exact backward pass.
// ---------------------------------------------------------------------
template <typename LossFn>
inline void finite_diff_gd_step(std::vector<double> &flat, LossFn loss_fn, double lr, double eps = 1e-4) {
  std::vector<double> grad(flat.size(), 0.0);
  for (size_t k = 0; k < flat.size(); ++k) {
    double orig = flat[k];
    flat[k] = orig + eps;
    double loss_plus = loss_fn(flat);
    flat[k] = orig - eps;
    double loss_minus = loss_fn(flat);
    flat[k] = orig;
    grad[k] = (loss_plus - loss_minus) / (2.0 * eps);
  }
  for (size_t k = 0; k < flat.size(); ++k) flat[k] -= lr * grad[k];
}

} // namespace sciml
