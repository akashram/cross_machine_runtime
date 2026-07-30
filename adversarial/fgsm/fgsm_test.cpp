// fgsm_test.cpp -- mechanics checks (the epsilon sweep + full
// accuracy-collapse measurement is step 4's job, not this file's):
//  1. every perturbed entry moves by EXACTLY +-epsilon (the L-infinity
//     bound FGSM is defined by) -- a single step in the sign direction,
//     not a scaled gradient step.
//  2. the perturbation direction is real gradient ASCENT: loss on the
//     perturbed batch is higher than on the clean batch, for a model that
//     was actually trained (a random/untrained model's loss surface has
//     no reason to behave this way locally, so this has to be checked
//     against a trained classifier to mean anything).
//  3. epsilon=0 is a no-op (perturbed == original exactly).
#include "fgsm.h"

#include <cmath>
#include <cstdio>
#include <random>

using namespace adversarial;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

float loss_at(const MLP &mlp, const Matrix &x, const std::vector<int> &labels) {
  Tensor xt(x);
  Tensor logits = mlp.forward(xt);
  return softmax_cross_entropy(logits, labels).value()(0, 0);
}

} // namespace

int main() {
  std::mt19937 data_rng(1);
  Dataset train = make_standard_dataset(/*per_class=*/40, data_rng);
  Dataset eval = make_standard_dataset(/*per_class=*/20, data_rng);

  std::mt19937 init_rng(2);
  MLP mlp = train_classifier(train, {2, 16, 3}, /*epochs=*/300, /*lr=*/0.1f, init_rng);

  constexpr float kEpsilon = 0.5f;
  Matrix perturbed = fgsm_attack(mlp, eval.x, eval.labels, kEpsilon);

  bool exact_linf_bound = true;
  for (int r = 0; r < eval.x.rows(); ++r)
    for (int c = 0; c < eval.x.cols(); ++c) {
      float delta = perturbed(r, c) - eval.x(r, c);
      if (std::abs(std::abs(delta) - kEpsilon) > 1e-5f) exact_linf_bound = false;
    }
  require(exact_linf_bound, "every perturbed entry moves by exactly +-epsilon");

  float clean_loss = loss_at(mlp, eval.x, eval.labels);
  float perturbed_loss = loss_at(mlp, perturbed, eval.labels);
  std::printf("  clean loss = %.4f, FGSM-perturbed loss (epsilon=%.2f) = %.4f\n", static_cast<double>(clean_loss),
              static_cast<double>(kEpsilon), static_cast<double>(perturbed_loss));
  require(perturbed_loss > clean_loss, "FGSM perturbation genuinely increases loss (real gradient ascent, on a trained model)");

  Matrix zero_eps = fgsm_attack(mlp, eval.x, eval.labels, 0.0f);
  bool identical = true;
  for (int r = 0; r < eval.x.rows(); ++r)
    for (int c = 0; c < eval.x.cols(); ++c)
      if (zero_eps(r, c) != eval.x(r, c)) identical = false;
  require(identical, "epsilon=0 is a no-op (perturbed input identical to the original)");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
