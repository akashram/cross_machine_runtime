#pragma once

// PLAN.md Phase 14 step 5: adversarial training (Madry et al. 2017's
// min-max formulation) -- replace each training batch with a
// PGD-perturbed version crafted against the CURRENT weights, then take a
// normal gradient-descent step on THAT batch. The inner maximization
// (PGD finds the worst-case perturbation within the epsilon-ball) and the
// outer minimization (SGD lowers loss on that worst case) alternate every
// epoch, exactly Madry et al.'s formulation.
//
// Reuses distributed_training/full_training_loop's forward -> backward ->
// grad-clip -> optimizer-step SHAPE, simplified to single-process: that
// step's grad_clipping.h clips a norm computed via ring_allreduce across
// SHARDED gradients (real multi-rank machinery this phase doesn't need --
// there is exactly one "rank" here, so the global norm IS the local
// norm). clip_grad_norm_local() below is the same "clip by norm to
// max_norm" rule with the distributed collective removed, not a
// different rule.

#include "../pgd/pgd.h"

#include <cmath>

namespace adversarial {

// Node::grad is a public member (see autograd.h) -- Tensor exposes
// mutable_value() for optimizer updates but no mutable_grad() equivalent,
// since normal training code only ever READS grad() and lets sgd_step()
// consume it. Clipping needs to WRITE a scaled grad back, so this reaches
// through Tensor::raw() (also public, by design, for exactly this kind of
// utility) rather than adding a mutable_grad() accessor whose only caller
// would be this one function.
inline void clip_grad_norm_local(const std::vector<Tensor> &params, float max_norm) {
  float sumsq = 0.0f;
  for (const auto &p : params) {
    const Matrix &g = p.grad();
    for (int r = 0; r < g.rows(); ++r)
      for (int c = 0; c < g.cols(); ++c) sumsq += g(r, c) * g(r, c);
  }
  float norm = std::sqrt(sumsq);
  constexpr float kEps = 1e-6f;
  float scale = std::min(1.0f, max_norm / (norm + kEps));
  if (scale >= 1.0f) return;
  for (const auto &p : params) p.raw()->grad = p.raw()->grad * scale;
}

struct AdversarialTrainingParams {
  std::vector<int> layer_dims;
  int epochs = 300;
  float lr = 0.1f;
  float epsilon = 1.0f;       // PGD attack budget used DURING training
  int pgd_num_steps = 10;
  float max_grad_norm = 5.0f; // <=0 disables clipping
};

inline MLP train_classifier_adversarial(const Dataset &train, const AdversarialTrainingParams &p, std::mt19937 &rng) {
  MLP mlp(p.layer_dims, rng);
  auto params = mlp.parameters();
  float pgd_step_size = std::max(p.epsilon / 4.0f, 0.02f);

  for (int epoch = 0; epoch < p.epochs; ++epoch) {
    // Inner maximization: the worst-case perturbation within the
    // epsilon-ball, against the model's CURRENT weights -- this is what
    // makes it adversarial TRAINING rather than training-then-attacking.
    Matrix adv_x = pgd_attack(mlp, train.x, train.labels, p.epsilon, pgd_step_size, p.pgd_num_steps);

    // Outer minimization: a normal gradient-descent step, but on the
    // adversarial batch instead of the clean one. pgd_attack's internal
    // input_gradient() calls already zero weight grads before each of
    // ITS steps, but that guarantee is documented as being about ITS OWN
    // calls, not this SEPARATE backward pass below -- zero explicitly
    // here too rather than relying on that as an implicit side effect.
    zero_grad(params);
    Tensor x(adv_x);
    Tensor logits = mlp.forward(x);
    Tensor loss = softmax_cross_entropy(logits, train.labels);
    loss.backward();
    if (p.max_grad_norm > 0.0f) clip_grad_norm_local(params, p.max_grad_norm);
    sgd_step(params, p.lr);
  }
  return mlp;
}

} // namespace adversarial
