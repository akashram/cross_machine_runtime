#pragma once

// PLAN.md Phase 14 step 8 (stretch goal): randomized smoothing (Cohen
// et al. 2019, "Certified Adversarial Robustness via Randomized
// Smoothing") -- a CERTIFIED (probabilistic-guarantee) defense,
// structurally different from PGD-based empirical defense (step 5):
// instead of training to survive a SPECIFIC attack, it wraps any base
// classifier f in a smoothed classifier g(x) = majority vote of f over
// many Gaussian-noised copies of x, then derives a PROVABLE robustness
// RADIUS from the vote margin -- a guarantee that holds against every
// possible attack within that radius, not just the ones tested.
//
// Real, disclosed scope reduction vs. the full paper: Cohen et al. use
// the EXACT Clopper-Pearson lower confidence bound (inverting the
// incomplete Beta distribution) on the top class's vote proportion. This
// implementation uses the normal (Wald) APPROXIMATION instead -- a real
// confidence bound, just a less precise one for small n or extreme
// proportions than the exact Beta-based interval. inverse_normal_cdf()
// itself is exact (bisection on std::erf, not an approximation formula);
// the approximation is specifically in using a normal distribution to
// model the binomial count instead of the exact Beta/binomial one.

#include "../pgd/pgd.h"

#include <cmath>
#include <random>
#include <vector>

namespace adversarial {

inline float standard_normal_cdf(float x) { return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f))); }

// Bisection on the (monotonic) CDF -- exact up to bisection tolerance,
// not a rational-approximation formula.
inline float inverse_normal_cdf(float p) {
  float lo = -10.0f, hi = 10.0f;
  for (int i = 0; i < 60; ++i) {
    float mid = 0.5f * (lo + hi);
    if (standard_normal_cdf(mid) < p) lo = mid; else hi = mid;
  }
  return 0.5f * (lo + hi);
}

struct SmoothedCertification {
  int predicted_class;
  float p_lower;          // normal-approximation lower confidence bound on the winning class's true vote probability
  float certified_radius; // sigma * Phi^-1(p_lower) if p_lower > 0.5, else 0 (abstain -- no certificate)
};

// `point_row` is a 1-row Matrix (a single example). Draws `n` Gaussian
// (stddev `sigma`) noised copies, classifies each with the base model,
// and majority-votes -- Cohen et al.'s g(x). `confidence` is the
// one-sided confidence level for the vote-proportion lower bound (e.g.
// 0.999 -- match this to how many points get certified vs. abstain when
// tuning n).
inline SmoothedCertification certify_smoothed(const MLP &mlp, const Matrix &point_row, float sigma, int n,
                                               std::mt19937 &rng, float confidence = 0.999f) {
  std::normal_distribution<float> noise(0.0f, sigma);
  std::vector<int> counts;
  for (int i = 0; i < n; ++i) {
    Matrix sample = point_row;
    for (int c = 0; c < sample.cols(); ++c) sample(0, c) += noise(rng);
    Tensor xt(sample);
    Tensor logits = mlp.forward(xt);
    if (counts.empty()) counts.assign(static_cast<std::size_t>(logits.cols()), 0);
    int pred = argmax_row(logits.value(), 0);
    ++counts[static_cast<std::size_t>(pred)];
  }

  int best_class = 0;
  for (std::size_t c = 1; c < counts.size(); ++c)
    if (counts[c] > counts[static_cast<std::size_t>(best_class)]) best_class = static_cast<int>(c);

  int k = counts[static_cast<std::size_t>(best_class)];
  float phat = static_cast<float>(k) / static_cast<float>(n);
  float se = std::sqrt(std::max(phat * (1.0f - phat), 0.0f) / static_cast<float>(n));
  float z = inverse_normal_cdf(confidence);
  float p_lower = phat - z * se;
  float radius = (p_lower > 0.5f) ? sigma * inverse_normal_cdf(p_lower) : 0.0f;
  return SmoothedCertification{best_class, p_lower, radius};
}

// Just the majority-vote prediction (the confidence bound / certified
// radius are irrelevant here, only the vote winner) -- what an empirical
// robustness comparison against PGD-perturbed inputs needs, one row at a
// time.
inline int smoothed_predict(const MLP &mlp, const Matrix &point_row, float sigma, int n, std::mt19937 &rng) {
  return certify_smoothed(mlp, point_row, sigma, n, rng).predicted_class;
}

} // namespace adversarial
