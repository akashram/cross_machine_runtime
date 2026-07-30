// pgd_test.cpp -- three checks:
//  1. the L-infinity budget is always respected: after any number of
//     steps, every perturbed entry stays within epsilon of the ORIGINAL
//     input (the projection actually projects).
//  2. structural relationship to FGSM: PGD with num_steps=1 and
//     step_size=epsilon must be EXACTLY FGSM at that epsilon (a single
//     full step lands exactly on the epsilon boundary, so the projection
//     is a no-op) -- verified as byte-identical output, not just "close".
//  3. PGD is at least as strong as a single FGSM step at the SAME total
//     epsilon budget: iterating with a smaller step_size finds a HIGHER
//     (or equal) loss than one big FGSM step, on a trained classifier --
//     the real, measured reason Madry et al. 2017 calls PGD the attack
//     "strong enough to trust a defense against."
#include "pgd.h"

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
  Matrix pgd_result = pgd_attack(mlp, eval.x, eval.labels, kEpsilon, /*step_size=*/0.1f, /*num_steps=*/10);

  bool within_budget = true;
  for (int r = 0; r < eval.x.rows(); ++r)
    for (int c = 0; c < eval.x.cols(); ++c)
      if (std::abs(pgd_result(r, c) - eval.x(r, c)) > kEpsilon + 1e-5f) within_budget = false;
  require(within_budget, "every perturbed entry stays within the L-infinity epsilon-ball around the original, after 10 iterations");

  Matrix pgd_one_step = pgd_attack(mlp, eval.x, eval.labels, kEpsilon, /*step_size=*/kEpsilon, /*num_steps=*/1);
  Matrix fgsm_result = fgsm_attack(mlp, eval.x, eval.labels, kEpsilon);
  bool identical_to_fgsm = true;
  for (int r = 0; r < eval.x.rows(); ++r)
    for (int c = 0; c < eval.x.cols(); ++c)
      if (pgd_one_step(r, c) != fgsm_result(r, c)) identical_to_fgsm = false;
  require(identical_to_fgsm, "PGD with num_steps=1, step_size=epsilon is EXACTLY FGSM (byte-identical output)");

  float fgsm_loss = loss_at(mlp, fgsm_result, eval.labels);
  float pgd_loss = loss_at(mlp, pgd_result, eval.labels);
  std::printf("  FGSM (1 step, epsilon=%.2f) loss = %.4f\n", static_cast<double>(kEpsilon), static_cast<double>(fgsm_loss));
  std::printf("  PGD (10 steps, step_size=0.1, epsilon=%.2f) loss = %.4f\n", static_cast<double>(kEpsilon),
              static_cast<double>(pgd_loss));
  require(pgd_loss >= fgsm_loss - 1e-6f, "PGD (iterative, same epsilon budget) finds a loss at least as high as a single FGSM step");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
