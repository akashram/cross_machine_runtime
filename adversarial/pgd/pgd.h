#pragma once

// PLAN.md Phase 14 step 3: PGD (Madry et al. 2017, "Towards Deep Learning
// Models Resistant to Adversarial Attacks") -- `num_steps` iterations of
// a smaller `step_size` in the sign-gradient direction, each followed by
// projecting back into the L-infinity `epsilon`-ball around the ORIGINAL
// input. Stronger than a single FGSM step because each iteration
// re-evaluates the gradient at the CURRENT perturbed point instead of
// trusting one linear approximation at the original point for the whole
// budget -- "the standard attack strong enough to trust a defense
// against" (PLAN.md's own framing, matching the literature).
//
// PGD with num_steps=1 and step_size=epsilon is EXACTLY FGSM (a single
// full-budget step, immediately at the epsilon boundary so projection is
// a no-op) -- verified directly in pgd_test.cpp rather than just noted
// here, since it's a real, checkable structural relationship between the
// two attacks.

#include "../fgsm/fgsm.h"

namespace adversarial {

inline float clip_scalar(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }

inline Matrix pgd_attack(const MLP &mlp, const Matrix &x_original, const std::vector<int> &labels, float epsilon,
                          float step_size, int num_steps) {
  Matrix x = x_original;
  for (int step = 0; step < num_steps; ++step) {
    Matrix grad = input_gradient(mlp, x, labels);
    for (int r = 0; r < x.rows(); ++r) {
      for (int c = 0; c < x.cols(); ++c) {
        float candidate = x(r, c) + step_size * sign(grad(r, c));
        x(r, c) = clip_scalar(candidate, x_original(r, c) - epsilon, x_original(r, c) + epsilon);
      }
    }
  }
  return x;
}

} // namespace adversarial
