// transferability_test.cpp -- PLAN.md Phase 14 step 7: do adversarial
// examples crafted against ONE model transfer to fool a differently-
// sized/differently-trained model? A real, measured cross-model transfer
// rate, contrasted against a random-noise baseline of the SAME magnitude
// (to rule out the trivial explanation that any perturbation this large
// fools any model, regardless of whether it's gradient-crafted).
//
// Model A: {2, 16, 3} (same architecture as every other step's model).
// Model B: {2, 32, 16, 3} -- wider AND one layer deeper, a different
// init seed, trained independently. Adversarial examples are crafted
// against A ONLY (B is never queried while crafting them) -- the whole
// point of a transfer study is B never sees the attack process.
#include "../pgd/pgd.h"

#include <cstdio>
#include <random>

using namespace adversarial;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

int predict(const MLP &mlp, const Matrix &x, int row) {
  Tensor xt(x);
  Tensor logits = mlp.forward(xt);
  return argmax_row(logits.value(), row);
}

} // namespace

int main() {
  std::mt19937 data_rng(1);
  Dataset train = make_standard_dataset(/*per_class=*/40, data_rng);
  Dataset eval = make_standard_dataset(/*per_class=*/30, data_rng);

  std::mt19937 rng_a(2);
  MLP model_a = train_classifier(train, {2, 16, 3}, /*epochs=*/300, /*lr=*/0.1f, rng_a);
  std::mt19937 rng_b(5); // different init seed
  MLP model_b = train_classifier(train, {2, 32, 16, 3}, /*epochs=*/300, /*lr=*/0.1f, rng_b); // different (wider, deeper) architecture

  float acc_a = accuracy(model_a, eval.x, eval.labels);
  float acc_b = accuracy(model_b, eval.x, eval.labels);
  std::printf("  model A {2,16,3} clean accuracy: %.3f\n", static_cast<double>(acc_a));
  std::printf("  model B {2,32,16,3} clean accuracy: %.3f\n", static_cast<double>(acc_b));

  constexpr float kEpsilon = 2.0f;
  constexpr float kStepSize = 0.5f;
  Matrix adv_x = pgd_attack(model_a, eval.x, eval.labels, kEpsilon, kStepSize, /*num_steps=*/10);

  int successful_attacks_on_a = 0, transferred_to_b = 0;
  for (int i = 0; i < eval.x.rows(); ++i) {
    int label = eval.labels[static_cast<std::size_t>(i)];
    if (predict(model_a, eval.x, i) != label) continue; // only count points A originally got right
    if (predict(model_a, adv_x, i) == label) continue;  // only count points the attack actually flips on A
    ++successful_attacks_on_a;
    if (predict(model_b, adv_x, i) != label) ++transferred_to_b;
  }
  double transfer_rate = successful_attacks_on_a > 0
                              ? static_cast<double>(transferred_to_b) / static_cast<double>(successful_attacks_on_a)
                              : 0.0;
  std::printf("  attacks against A that succeeded: %d/%d\n", successful_attacks_on_a, eval.x.rows());
  std::printf("  of those, also fooled B (transfer rate): %d/%d = %.3f\n", transferred_to_b, successful_attacks_on_a,
              transfer_rate);
  require(successful_attacks_on_a > 0, "the PGD attack against model A actually succeeds on at least some points (a meaningful transfer measurement needs successful attacks to transfer)");

  // Baseline: uniform RANDOM perturbation of the same L-infinity
  // magnitude, not gradient-crafted -- rules out "any perturbation this
  // size fools any model" as the explanation for a nonzero transfer rate.
  std::mt19937 noise_rng(7);
  std::uniform_real_distribution<float> noise(-kEpsilon, kEpsilon);
  Matrix random_x = eval.x;
  for (int r = 0; r < random_x.rows(); ++r)
    for (int c = 0; c < random_x.cols(); ++c) random_x(r, c) += noise(noise_rng);

  int random_fools_b = 0, random_eligible = 0;
  for (int i = 0; i < eval.x.rows(); ++i) {
    int label = eval.labels[static_cast<std::size_t>(i)];
    if (predict(model_b, eval.x, i) != label) continue;
    ++random_eligible;
    if (predict(model_b, random_x, i) != label) ++random_fools_b;
  }
  double random_fool_rate =
      random_eligible > 0 ? static_cast<double>(random_fools_b) / static_cast<double>(random_eligible) : 0.0;
  std::printf("  random noise (same epsilon) fools B: %d/%d = %.3f\n", random_fools_b, random_eligible, random_fool_rate);

  require(transfer_rate > random_fool_rate,
          "gradient-crafted adversarial examples transfer to model B at a higher rate than same-magnitude random noise fools it (genuine transfer, not just 'any big enough perturbation confuses any model')");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
