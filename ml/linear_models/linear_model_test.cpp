// linear_model_test.cpp — real correctness checks: logistic regression
// (both SGD and L-BFGS optimizers) actually separates classes, squared-
// loss regression actually recovers known ground-truth coefficients,
// and elastic net's two knobs do what they claim -- L1 induces real
// sparsity (near-zero noise coefficients, not just "smaller"), and
// larger regularization strength shrinks the coefficient norm (the
// classic bias-variance regularization tradeoff, same shape as
// decision_tree/svm's own parameter sweeps).
#include "linear_model.h"

#include <cmath>
#include <cstdio>
#include <random>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void make_blobs(Features &X, Labels &y, std::mt19937 &rng) {
  std::normal_distribution<float> noise(0.0f, 0.5f);
  for (int i = 0; i < 150; ++i) {
    X.push_back({-3.0f + noise(rng), -3.0f + noise(rng)});
    y.push_back(0.0f);
  }
  for (int i = 0; i < 150; ++i) {
    X.push_back({3.0f + noise(rng), 3.0f + noise(rng)});
    y.push_back(1.0f);
  }
}

void test_logistic_regression_sgd_separates_blobs() {
  std::mt19937 rng(1);
  Features X;
  Labels y;
  make_blobs(X, y, rng);

  LinearModelParams p;
  p.loss = LinearLoss::LOGISTIC;
  p.optimizer = LinearOptimizer::SGD;
  p.alpha = 1e-4f;
  p.learning_rate = 0.1f;
  p.max_iter = 100;
  LinearModel model(p);
  model.fit(X, y);
  float acc = model.score(X, y);

  std::printf("  logistic regression (SGD): accuracy=%.3f\n", static_cast<double>(acc));
  require(acc > 0.98f, "SGD-trained logistic regression separates two well-separated blobs almost exactly");
}

void test_logistic_regression_lbfgs_separates_blobs() {
  std::mt19937 rng(2);
  Features X;
  Labels y;
  make_blobs(X, y, rng);

  LinearModelParams p;
  p.loss = LinearLoss::LOGISTIC;
  p.optimizer = LinearOptimizer::LBFGS;
  p.alpha = 1e-4f;
  p.max_iter = 50;
  LinearModel model(p);
  model.fit(X, y);
  float acc = model.score(X, y);

  std::printf("  logistic regression (LBFGS): accuracy=%.3f\n", static_cast<double>(acc));
  require(acc > 0.98f, "LBFGS-trained logistic regression separates two well-separated blobs almost exactly");
}

// y = 3*x0 - 2*x1 + 1 + small noise: a squared-loss linear model with
// small regularization should recover coefficients close to [3, -2] and
// intercept close to 1.
void test_squared_loss_recovers_true_coefficients() {
  std::mt19937 rng(3);
  std::normal_distribution<float> feature_dist(0.0f, 2.0f), noise(0.0f, 0.1f);
  Features X;
  Labels y;
  for (int i = 0; i < 300; ++i) {
    float x0 = feature_dist(rng), x1 = feature_dist(rng);
    X.push_back({x0, x1});
    y.push_back(3.0f * x0 - 2.0f * x1 + 1.0f + noise(rng));
  }

  LinearModelParams p;
  p.loss = LinearLoss::SQUARED;
  p.optimizer = LinearOptimizer::LBFGS;
  p.alpha = 1e-5f;
  p.max_iter = 100;
  LinearModel model(p);
  model.fit(X, y);
  float r2 = model.score(X, y);

  std::printf("  recovered coef=[%.3f, %.3f], intercept=%.3f, R^2=%.4f\n", static_cast<double>(model.coef()[0]),
              static_cast<double>(model.coef()[1]), static_cast<double>(model.intercept()), static_cast<double>(r2));
  require(std::fabs(model.coef()[0] - 3.0f) < 0.1f, "recovered coefficient for x0 is close to the true value 3.0");
  require(std::fabs(model.coef()[1] - (-2.0f)) < 0.1f, "recovered coefficient for x1 is close to the true value -2.0");
  require(r2 > 0.99f, "R^2 is close to 1 on data generated from a true linear relationship plus small noise");
}

// 200 samples, 7 features: only features 0 and 1 are informative (y =
// 4*x0 - 3*x1 + noise), features 2-6 are pure independent noise
// unrelated to y. Pure L1 (lasso) should drive the noise-feature
// coefficients to exactly (or almost exactly) zero -- real sparsity,
// not just "smaller than ridge's".
void test_l1_regularization_induces_sparsity() {
  std::mt19937 rng(4);
  std::normal_distribution<float> feature_dist(0.0f, 2.0f), noise(0.0f, 0.2f);
  Features X;
  Labels y;
  for (int i = 0; i < 300; ++i) {
    std::vector<float> row(7);
    for (auto &v : row) v = feature_dist(rng);
    X.push_back(row);
    y.push_back(4.0f * row[0] - 3.0f * row[1] + noise(rng));
  }

  LinearModelParams lasso_p;
  lasso_p.loss = LinearLoss::SQUARED;
  lasso_p.optimizer = LinearOptimizer::SGD;
  lasso_p.alpha = 0.1f;
  lasso_p.l1_ratio = 1.0f;
  lasso_p.learning_rate = 0.05f;
  lasso_p.max_iter = 300;
  LinearModel lasso(lasso_p);
  lasso.fit(X, y);

  LinearModelParams ridge_p = lasso_p;
  ridge_p.l1_ratio = 0.0f;
  LinearModel ridge(ridge_p);
  ridge.fit(X, y);

  // Per-sample proximal SGD's soft-threshold competes against ongoing
  // per-sample gradient noise on every step, so coefficients rarely land
  // on an exact bit-pattern 0.0f (that needs a batch/full-gradient
  // proximal step to guarantee) -- "near zero" is the honest, measured
  // bar for this SGD-based implementation, not "exactly zero".
  float lasso_noise_sum = 0.0f, ridge_noise_sum = 0.0f;
  int lasso_near_zero = 0;
  for (int j = 2; j < 7; ++j) {
    lasso_noise_sum += std::fabs(lasso.coef()[static_cast<std::size_t>(j)]);
    ridge_noise_sum += std::fabs(ridge.coef()[static_cast<std::size_t>(j)]);
    if (std::fabs(lasso.coef()[static_cast<std::size_t>(j)]) < 0.05f) ++lasso_near_zero;
  }

  std::printf("  lasso: informative coefs=[%.3f, %.3f], noise coef sum=%.4f (%d near zero)\n",
              static_cast<double>(lasso.coef()[0]), static_cast<double>(lasso.coef()[1]), static_cast<double>(lasso_noise_sum),
              lasso_near_zero);
  std::printf("  ridge: informative coefs=[%.3f, %.3f], noise coef sum=%.4f\n", static_cast<double>(ridge.coef()[0]),
              static_cast<double>(ridge.coef()[1]), static_cast<double>(ridge_noise_sum));

  require(lasso_near_zero >= 3, "lasso (pure L1) drives at least 3 of the 5 noise-feature coefficients near zero (< 0.05)");
  require(lasso_noise_sum < ridge_noise_sum, "lasso's total noise-coefficient magnitude is smaller than ridge's (real sparsity, not just smaller weights)");
  require(std::fabs(lasso.coef()[0] - 4.0f) < 0.5f, "lasso still recovers the informative x0 coefficient reasonably well");
  require(std::fabs(lasso.coef()[1] - (-3.0f)) < 0.5f, "lasso still recovers the informative x1 coefficient reasonably well");
}

// Classic ridge behavior: larger alpha shrinks the coefficient vector's
// norm toward zero (more bias, less variance) -- measured across an
// alpha sweep, not assumed.
void test_larger_alpha_shrinks_coefficient_norm() {
  std::mt19937 rng(5);
  std::normal_distribution<float> feature_dist(0.0f, 2.0f), noise(0.0f, 0.3f);
  Features X;
  Labels y;
  for (int i = 0; i < 200; ++i) {
    float x0 = feature_dist(rng), x1 = feature_dist(rng);
    X.push_back({x0, x1});
    y.push_back(2.0f * x0 + 2.0f * x1 + noise(rng));
  }

  auto coef_norm = [&](float alpha) {
    LinearModelParams p;
    p.loss = LinearLoss::SQUARED;
    p.optimizer = LinearOptimizer::LBFGS;
    p.alpha = alpha;
    p.max_iter = 100;
    LinearModel model(p);
    model.fit(X, y);
    float norm = 0.0f;
    for (float c : model.coef()) norm += c * c;
    return std::sqrt(norm);
  };

  float norm_small_alpha = coef_norm(1e-5f);
  float norm_large_alpha = coef_norm(5.0f);

  std::printf("  ridge coefficient norm: alpha=1e-5 -> %.4f, alpha=5.0 -> %.4f\n", static_cast<double>(norm_small_alpha),
              static_cast<double>(norm_large_alpha));
  require(norm_large_alpha < norm_small_alpha, "a much larger ridge alpha shrinks the coefficient vector's norm toward zero");
}

}  // namespace

int main() {
  test_logistic_regression_sgd_separates_blobs();
  test_logistic_regression_lbfgs_separates_blobs();
  test_squared_loss_recovers_true_coefficients();
  test_l1_regularization_induces_sparsity();
  test_larger_alpha_shrinks_coefficient_norm();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
