#pragma once

// PLAN.md Phase 14 step 1: expose gradients of loss with respect to the
// INPUT, not just weights.
//
// The generic reverse-mode tape (distributed_training/autograd/autograd.h)
// does not special-case parameter Tensors vs. input Tensors anywhere --
// every op's backward_fn adds its contribution into EVERY parent Node's
// grad field uniformly. distributed_training/full_training_loop already
// wraps its input batch in a Tensor (`Tensor x(shard_x);`) before calling
// mlp.forward(x), so x.grad() already holds d(loss)/d(x) after
// loss.backward() -- nothing in the engine needs to change. What this
// file adds is making that a first-class, TESTED capability for
// adversarial attack code (steps 2/3), not an incidental byproduct never
// explicitly relied on anywhere else in the codebase.
//
// One real correctness detail this function handles: repeated calls (PGD,
// step 3, calls this once per iteration) must not let the MODEL's WEIGHT
// gradients silently accumulate across calls -- Tensor::backward() only
// SEEDS the output node's grad, it does not zero every ancestor's grad
// first (add_inplace onto whatever's already there is deliberate, so a
// Node used by two consumers sums both contributions -- see autograd.h's
// own design note). Since attack code never wants the weight grads, this
// function zeros them before every call, making each call self-contained
// regardless of how many times it's been called before, and regardless of
// what the caller does with the model's weight Tensors afterward (e.g.
// step 5's adversarial training loop calls this INSIDE its own training
// step to craft a PGD example, then must zero_grad() again itself before
// its own real backward pass -- documented at that call site).

#include "../task/classification_task.h"

namespace adversarial {

// d(loss)/d(x), same shape as x. Read-only w.r.t. `mlp`'s weights (they
// get a zeroed, then discarded, gradient as an unavoidable byproduct of
// running the shared forward graph through softmax_cross_entropy's
// backward -- see the file comment above).
inline Matrix input_gradient(const MLP &mlp, const Matrix &x, const std::vector<int> &labels) {
  auto params = mlp.parameters();
  zero_grad(params);
  Tensor xt(x);
  Tensor logits = mlp.forward(xt);
  Tensor loss = softmax_cross_entropy(logits, labels);
  loss.backward();
  return xt.grad();
}

} // namespace adversarial
