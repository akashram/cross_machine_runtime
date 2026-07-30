// robustness_tradeoff_test.cpp -- PLAN.md Phase 14 step 6: measure clean
// accuracy of the adversarially-trained model vs. the undefended model,
// AND robust accuracy under attack for both, across a SWEEP of training
// epsilons -- not just the single epsilon step 5 checked. Tsipras et al.
// 2018's "Robustness May Be at Odds with Accuracy" is a real, well-
// documented phenomenon: this measures whether it actually shows up
// here, rather than citing the paper and assuming it does.
//
// Pure composition of step 4 (vulnerability_measurement) and step 5
// (adversarial_training) -- no new attack or training logic, only the
// sweep-across-training-epsilon comparison itself is new.
#include "../adversarial_training/adversarial_training.h"
#include "../vulnerability_measurement/vulnerability_measurement.h"

#include <cstdio>
#include <random>

using namespace adversarial;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

} // namespace

int main() {
  std::mt19937 data_rng(1);
  Dataset train = make_standard_dataset(/*per_class=*/40, data_rng);
  Dataset eval = make_standard_dataset(/*per_class=*/30, data_rng);

  std::mt19937 undefended_rng(2);
  MLP undefended = train_classifier(train, {2, 16, 3}, /*epochs=*/300, /*lr=*/0.1f, undefended_rng);
  float undefended_clean = accuracy(undefended, eval.x, eval.labels);
  std::printf("  undefended clean accuracy: %.3f\n", static_cast<double>(undefended_clean));

  std::vector<float> training_epsilons{0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> undefended_robust;
  for (float eps : training_epsilons) {
    float step_size = std::max(eps / 4.0f, 0.02f);
    Matrix pgd_x = pgd_attack(undefended, eval.x, eval.labels, eps, step_size, /*num_steps=*/10);
    undefended_robust.push_back(accuracy(undefended, pgd_x, eval.labels));
  }

  std::printf("  %-12s %-14s %-16s %-14s %-16s\n", "epsilon", "undef. clean", "undef. robust", "adv. clean", "adv. robust");

  std::vector<float> adv_clean, adv_robust;
  for (std::size_t i = 0; i < training_epsilons.size(); ++i) {
    float eps = training_epsilons[i];
    AdversarialTrainingParams p;
    p.layer_dims = {2, 16, 3};
    p.epochs = 300;
    p.lr = 0.1f;
    p.epsilon = eps;
    p.pgd_num_steps = 10;
    std::mt19937 adv_rng(2);
    MLP adv_trained = train_classifier_adversarial(train, p, adv_rng);

    float clean = accuracy(adv_trained, eval.x, eval.labels);
    float step_size = std::max(eps / 4.0f, 0.02f);
    Matrix pgd_x = pgd_attack(adv_trained, eval.x, eval.labels, eps, step_size, /*num_steps=*/10);
    float robust = accuracy(adv_trained, pgd_x, eval.labels);
    adv_clean.push_back(clean);
    adv_robust.push_back(robust);

    std::printf("  %-12.2f %-14.3f %-16.3f %-14.3f %-16.3f\n", static_cast<double>(eps),
                static_cast<double>(undefended_clean), static_cast<double>(undefended_robust[i]),
                static_cast<double>(clean), static_cast<double>(robust));
  }

  bool robust_improves_everywhere = true;
  for (std::size_t i = 0; i < training_epsilons.size(); ++i)
    if (adv_robust[i] < undefended_robust[i] - 1e-6f) robust_improves_everywhere = false;
  require(robust_improves_everywhere, "adversarial training's robust accuracy is at least the undefended model's, at every training epsilon in the sweep");

  float clean_at_largest_eps = adv_clean.back();
  std::printf("  clean-accuracy cost at the largest training epsilon (%.2f): undefended=%.3f -> adv-trained=%.3f\n",
              static_cast<double>(training_epsilons.back()), static_cast<double>(undefended_clean),
              static_cast<double>(clean_at_largest_eps));
  if (clean_at_largest_eps < undefended_clean - 0.02f)
    std::printf("  finding: a real clean-accuracy cost shows up at this training epsilon (Tsipras et al. 2018, measured not assumed)\n");
  else
    std::printf("  finding: no measurable clean-accuracy cost at this training epsilon on this task -- reported honestly, not forced\n");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
