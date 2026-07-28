// svm_test.cpp — real correctness checks for simplified-SMO SVM, not
// just "it runs": a linearly separable dataset solved exactly by a
// linear kernel with FEW support vectors (not every point), the kernel
// trick actually mattering (RBF solves XOR, linear cannot), and C's
// regularization effect (larger C fits noisy training data more
// tightly, the classic margin-vs-violation tradeoff).
#include "svm.h"

#include <cstdio>
#include <random>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void make_xor_dataset(int n, Features &X, Labels &y, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  X.clear();
  y.clear();
  for (int i = 0; i < n; ++i) {
    float x0 = dist(rng), x1 = dist(rng);
    X.push_back({x0, x1});
    bool label = (x0 > 0) != (x1 > 0);
    y.push_back(label ? 1.0f : -1.0f);
  }
}

// Two well-separated 2D Gaussian blobs -- a linear kernel should solve
// this exactly, and (Platt's whole point) only the points near the
// boundary should end up as support vectors, not the whole dataset.
void test_linear_kernel_separates_blobs_with_few_support_vectors() {
  std::mt19937 rng(1);
  std::normal_distribution<float> noise(0.0f, 0.3f);
  Features X;
  Labels y;
  for (int i = 0; i < 100; ++i) {
    X.push_back({-2.0f + noise(rng), -2.0f + noise(rng)});
    y.push_back(-1.0f);
  }
  for (int i = 0; i < 100; ++i) {
    X.push_back({2.0f + noise(rng), 2.0f + noise(rng)});
    y.push_back(1.0f);
  }

  SVMParams p;
  p.kernel = KernelType::LINEAR;
  p.C = 1.0f;
  SVM svm(p);
  svm.fit(X, y);
  float acc = svm.score(X, y);

  std::printf("  linear-kernel blobs: train accuracy=%.3f, support vectors=%d / %d\n", static_cast<double>(acc),
              svm.n_support_vectors(), static_cast<int>(X.size()));
  require(acc > 0.98f, "a linear kernel separates two well-separated blobs almost exactly");
  require(svm.n_support_vectors() < static_cast<int>(X.size()) / 4,
          "only points near the margin become support vectors, not the whole dataset");
}

// The kernel trick's entire point: XOR has no linear decision boundary
// (same reason decision_tree_test.cpp's depth-1 stump fails on it), so a
// linear-kernel SVM should stay near chance while an RBF kernel -- which
// implicitly maps into an infinite-dimensional feature space -- solves
// it well.
void test_rbf_kernel_solves_xor_linear_cannot() {
  std::mt19937 rng(2);
  Features X;
  Labels y;
  make_xor_dataset(200, X, y, rng);

  SVMParams linear_p;
  linear_p.kernel = KernelType::LINEAR;
  SVM linear_svm(linear_p);
  linear_svm.fit(X, y);
  float linear_acc = linear_svm.score(X, y);

  SVMParams rbf_p;
  rbf_p.kernel = KernelType::RBF;
  rbf_p.gamma = 2.0f;
  rbf_p.C = 10.0f;
  SVM rbf_svm(rbf_p);
  rbf_svm.fit(X, y);
  float rbf_acc = rbf_svm.score(X, y);

  std::printf("  XOR train accuracy: linear kernel=%.3f, RBF kernel=%.3f\n", static_cast<double>(linear_acc),
              static_cast<double>(rbf_acc));
  require(linear_acc < 0.7f, "a linear kernel cannot solve XOR (no linear boundary separates the classes)");
  require(rbf_acc > 0.95f, "an RBF kernel solves XOR almost exactly via the kernel trick");
}

// Larger C penalizes margin violations more, so an SVM should fit noisy
// training data more tightly (higher train accuracy) at large C than at
// small C, which prioritizes a wide margin over classifying every point
// correctly -- the classic C bias/variance tradeoff.
void test_larger_C_fits_noisy_data_more_tightly() {
  std::mt19937 rng(3);
  std::normal_distribution<float> noise(0.0f, 0.3f);
  std::uniform_real_distribution<float> flip_roll(0.0f, 1.0f);
  Features X;
  Labels y;
  for (int i = 0; i < 80; ++i) {
    X.push_back({-2.0f + noise(rng), -2.0f + noise(rng)});
    y.push_back(flip_roll(rng) < 0.15f ? 1.0f : -1.0f);
  }
  for (int i = 0; i < 80; ++i) {
    X.push_back({2.0f + noise(rng), 2.0f + noise(rng)});
    y.push_back(flip_roll(rng) < 0.15f ? -1.0f : 1.0f);
  }

  SVMParams loose_p;
  loose_p.kernel = KernelType::LINEAR;
  loose_p.C = 0.01f;
  SVM loose_svm(loose_p);
  loose_svm.fit(X, y);
  float loose_acc = loose_svm.score(X, y);

  SVMParams strict_p;
  strict_p.kernel = KernelType::LINEAR;
  strict_p.C = 100.0f;
  SVM strict_svm(strict_p);
  strict_svm.fit(X, y);
  float strict_acc = strict_svm.score(X, y);

  std::printf("  noisy-blob train accuracy: C=0.01 -> %.3f, C=100 -> %.3f\n", static_cast<double>(loose_acc),
              static_cast<double>(strict_acc));
  require(strict_acc >= loose_acc, "a larger C fits noisy training data at least as tightly as a small C (less margin, fewer violations tolerated)");
}

}  // namespace

int main() {
  test_linear_kernel_separates_blobs_with_few_support_vectors();
  test_rbf_kernel_solves_xor_linear_cannot();
  test_larger_C_fits_noisy_data_more_tightly();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
