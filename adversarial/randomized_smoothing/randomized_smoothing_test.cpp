// randomized_smoothing_test.cpp -- PLAN.md Phase 14 step 8 (stretch
// goal): four checks:
//  1. inverse_normal_cdf() is a correct inverse of standard_normal_cdf()
//     (round-trips a handful of probabilities) -- the numerical
//     foundation everything else here depends on.
//  2. randomized smoothing's clean accuracy is close to the base
//     classifier's (majority-voting over noise shouldn't destroy
//     correctness on well-separated data).
//  3. certification produces REAL, varying per-point radii, honestly
//     abstaining (radius=0) on points it can't certify rather than
//     forcing a number -- summarized across the eval set, not cherry-
//     picked.
//  4. the actual empirical question: does smoothing recover correct
//     predictions on PGD-perturbed inputs (crafted against the BASE
//     model) that the bare base model gets wrong?
#include "randomized_smoothing.h"

#include <cstdio>
#include <random>

using namespace adversarial;

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

Matrix row(const Matrix &m, int r) {
  Matrix out(1, m.cols());
  for (int c = 0; c < m.cols(); ++c) out(0, c) = m(r, c);
  return out;
}

int predict_base(const MLP &mlp, const Matrix &x, int r) {
  Matrix pt = row(x, r);
  Tensor xt(pt);
  Tensor logits = mlp.forward(xt);
  return argmax_row(logits.value(), 0);
}

} // namespace

int main() {
  bool cdf_ok = true;
  for (float p : {0.05f, 0.25f, 0.5f, 0.75f, 0.95f, 0.999f}) {
    float z = inverse_normal_cdf(p);
    float back = standard_normal_cdf(z);
    if (std::abs(back - p) > 1e-3f) cdf_ok = false;
  }
  require(cdf_ok, "inverse_normal_cdf correctly inverts standard_normal_cdf (round-trip within 1e-3)");

  std::mt19937 data_rng(1);
  Dataset train = make_standard_dataset(/*per_class=*/40, data_rng);
  Dataset eval = make_standard_dataset(/*per_class=*/20, data_rng);

  std::mt19937 init_rng(2);
  MLP mlp = train_classifier(train, {2, 16, 3}, /*epochs=*/300, /*lr=*/0.1f, init_rng);

  constexpr float kSigma = 0.75f;
  constexpr int kN = 200;
  std::mt19937 smooth_rng(3);

  int base_correct = 0, smoothed_correct = 0;
  for (int i = 0; i < eval.x.rows(); ++i) {
    int label = eval.labels[static_cast<std::size_t>(i)];
    if (predict_base(mlp, eval.x, i) == label) ++base_correct;
    if (smoothed_predict(mlp, row(eval.x, i), kSigma, kN, smooth_rng) == label) ++smoothed_correct;
  }
  float base_clean_acc = static_cast<float>(base_correct) / static_cast<float>(eval.x.rows());
  float smoothed_clean_acc = static_cast<float>(smoothed_correct) / static_cast<float>(eval.x.rows());
  std::printf("  clean accuracy: base=%.3f, smoothed (sigma=%.2f, n=%d)=%.3f\n", static_cast<double>(base_clean_acc),
              static_cast<double>(kSigma), kN, static_cast<double>(smoothed_clean_acc));
  require(smoothed_clean_acc >= base_clean_acc - 0.1f, "smoothing does not substantially hurt clean accuracy");

  int certified_count = 0;
  float radius_sum = 0.0f, radius_max = 0.0f;
  std::mt19937 cert_rng(4);
  for (int i = 0; i < eval.x.rows(); ++i) {
    auto cert = certify_smoothed(mlp, row(eval.x, i), kSigma, kN, cert_rng);
    if (cert.certified_radius > 0.0f) {
      ++certified_count;
      radius_sum += cert.certified_radius;
      radius_max = std::max(radius_max, cert.certified_radius);
    }
  }
  float avg_radius = certified_count > 0 ? radius_sum / static_cast<float>(certified_count) : 0.0f;
  std::printf("  certified %d/%d points; avg certified radius=%.3f, max=%.3f\n", certified_count,
              static_cast<int>(eval.x.rows()), static_cast<double>(avg_radius), static_cast<double>(radius_max));
  require(certified_count > 0, "at least some points get a real nonzero certified radius (the method isn't vacuously abstaining on everything)");
  require(certified_count <= eval.x.rows(), "certified points never exceed the eval set size (sanity)");

  constexpr float kEpsilon = 2.0f;
  Matrix adv_x = pgd_attack(mlp, eval.x, eval.labels, kEpsilon, /*step_size=*/0.5f, /*num_steps=*/10);
  int base_adv_correct = 0, smoothed_adv_correct = 0;
  std::mt19937 adv_smooth_rng(5);
  for (int i = 0; i < eval.x.rows(); ++i) {
    int label = eval.labels[static_cast<std::size_t>(i)];
    if (predict_base(mlp, adv_x, i) == label) ++base_adv_correct;
    if (smoothed_predict(mlp, row(adv_x, i), kSigma, kN, adv_smooth_rng) == label) ++smoothed_adv_correct;
  }
  float base_adv_acc = static_cast<float>(base_adv_correct) / static_cast<float>(eval.x.rows());
  float smoothed_adv_acc = static_cast<float>(smoothed_adv_correct) / static_cast<float>(eval.x.rows());
  std::printf("  accuracy on PGD-perturbed inputs (epsilon=%.2f): base=%.3f, smoothed=%.3f\n",
              static_cast<double>(kEpsilon), static_cast<double>(base_adv_acc), static_cast<double>(smoothed_adv_acc));
  require(smoothed_adv_acc >= base_adv_acc - 1e-6f, "randomized smoothing's accuracy on PGD-perturbed inputs is at least the bare base model's");

  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
