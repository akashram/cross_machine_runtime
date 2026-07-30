// adversarial_training_test.cpp -- three checks:
//  1. clip_grad_norm_local() actually bounds the norm (a direct unit
//     check of the clipping function itself).
//  2. the adversarially-trained model still reaches reasonable clean
//     accuracy -- training on PGD-perturbed batches the whole time has to
//     actually converge, not just "run without crashing."
//  3. the core claim: at the SAME attack epsilon used during training,
//     the adversarially-trained model has measurably higher robust
//     accuracy than an undefended model trained the normal way -- the
//     defense actually defends. (The full robustness/accuracy TRADEOFF
//     curve, including clean-accuracy cost, is step 6's job.)
#include "adversarial_training.h"

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

} // namespace

int main() {
  std::mt19937 clip_rng(9);
  Dataset clip_ds = make_standard_dataset(/*per_class=*/10, clip_rng);
  MLP clip_mlp({2, 8, 3}, clip_rng);
  auto clip_params = clip_mlp.parameters();
  {
    Tensor x(clip_ds.x);
    Tensor logits = clip_mlp.forward(x);
    Tensor loss = softmax_cross_entropy(logits, clip_ds.labels);
    loss.backward();
  }
  auto grad_norm = [](const std::vector<Tensor> &params) {
    float sumsq = 0.0f;
    for (const auto &p : params) {
      const Matrix &g = p.grad();
      for (int r = 0; r < g.rows(); ++r)
        for (int c = 0; c < g.cols(); ++c) sumsq += g(r, c) * g(r, c);
    }
    return std::sqrt(sumsq);
  };
  float norm_before = grad_norm(clip_params);
  constexpr float kMaxNorm = 0.1f;
  clip_grad_norm_local(clip_params, kMaxNorm);
  float norm_after = grad_norm(clip_params);
  std::printf("  grad norm before clip: %.4f, after clip (max_norm=%.2f): %.4f\n", static_cast<double>(norm_before),
              static_cast<double>(kMaxNorm), static_cast<double>(norm_after));
  require(norm_after <= kMaxNorm + 1e-4f, "clip_grad_norm_local bounds the gradient norm to max_norm");

  std::mt19937 data_rng(1);
  Dataset train = make_standard_dataset(/*per_class=*/40, data_rng);
  Dataset eval = make_standard_dataset(/*per_class=*/30, data_rng);

  std::mt19937 undefended_rng(2);
  MLP undefended = train_classifier(train, {2, 16, 3}, /*epochs=*/300, /*lr=*/0.1f, undefended_rng);

  AdversarialTrainingParams adv_params;
  adv_params.layer_dims = {2, 16, 3};
  adv_params.epochs = 300;
  adv_params.lr = 0.1f;
  adv_params.epsilon = 1.5f;
  adv_params.pgd_num_steps = 10;
  std::mt19937 adv_rng(2);
  MLP adv_trained = train_classifier_adversarial(train, adv_params, adv_rng);

  float adv_clean_accuracy = accuracy(adv_trained, eval.x, eval.labels);
  std::printf("  adversarially-trained model's clean accuracy: %.3f\n", static_cast<double>(adv_clean_accuracy));
  require(adv_clean_accuracy >= 0.7f, "the adversarially-trained model still reaches reasonable clean accuracy (training actually converged)");

  Matrix pgd_x_undefended = pgd_attack(undefended, eval.x, eval.labels, adv_params.epsilon,
                                        std::max(adv_params.epsilon / 4.0f, 0.02f), adv_params.pgd_num_steps);
  Matrix pgd_x_adv = pgd_attack(adv_trained, eval.x, eval.labels, adv_params.epsilon,
                                 std::max(adv_params.epsilon / 4.0f, 0.02f), adv_params.pgd_num_steps);
  float robust_accuracy_undefended = accuracy(undefended, pgd_x_undefended, eval.labels);
  float robust_accuracy_adv = accuracy(adv_trained, pgd_x_adv, eval.labels);
  std::printf("  robust accuracy under PGD (epsilon=%.2f): undefended=%.3f, adversarially-trained=%.3f\n",
              static_cast<double>(adv_params.epsilon), static_cast<double>(robust_accuracy_undefended),
              static_cast<double>(robust_accuracy_adv));
  require(robust_accuracy_adv > robust_accuracy_undefended,
          "the adversarially-trained model has measurably higher robust accuracy than the undefended model, at the SAME attack epsilon");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
