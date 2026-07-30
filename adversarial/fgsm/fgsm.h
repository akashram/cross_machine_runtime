#pragma once

// PLAN.md Phase 14 step 2: FGSM (Goodfellow et al. 2014, "Explaining and
// Harnessing Adversarial Examples") -- perturb the input by
// `epsilon * sign(gradient)` to MAXIMIZE loss (gradient ASCENT, the
// opposite direction from training's gradient descent), using step 1's
// input_gradient(). A single step, unlike PGD's (step 3) iterative
// refinement -- the classic "cheap, one-shot" attack.

#include "../input_gradients/input_gradients.h"

namespace adversarial {

inline float sign(float v) { return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); }

inline Matrix fgsm_attack(const MLP &mlp, const Matrix &x, const std::vector<int> &labels, float epsilon) {
  Matrix grad = input_gradient(mlp, x, labels);
  Matrix perturbed = x;
  for (int r = 0; r < x.rows(); ++r)
    for (int c = 0; c < x.cols(); ++c) perturbed(r, c) += epsilon * sign(grad(r, c));
  return perturbed;
}

} // namespace adversarial
