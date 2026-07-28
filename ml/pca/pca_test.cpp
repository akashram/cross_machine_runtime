// pca_test.cpp — real correctness checks for randomized-SVD PCA: the
// standard SVD invariants (orthonormal components, explained variance
// ratios summing to ~1 at full rank), a known low-rank-embedded dataset
// actually gets reconstructed well from just its top components, a
// clear variance-along-an-axis dataset actually gets its first
// component aligned to that axis, and whitening actually produces
// unit-variance transformed features.
#include "pca.h"

#include <cmath>
#include <cstdio>
#include <random>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

float dot(const std::vector<float> &a, const std::vector<float> &b) {
  float s = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

// 200 points in 5D where only the first two dimensions carry real
// signal (std 5 and 3) and the remaining three are near-zero noise
// (std 0.1) -- a genuine low-rank-embedded dataset, not just random
// noise, so a good PCA should recover it.
void test_pca_reconstructs_low_rank_embedded_data() {
  std::mt19937 rng(1);
  std::normal_distribution<float> signal1(0.0f, 5.0f), signal2(0.0f, 3.0f), noise(0.0f, 0.1f);
  Features X;
  for (int i = 0; i < 200; ++i) X.push_back({signal1(rng), signal2(rng), noise(rng), noise(rng), noise(rng)});

  PCAParams p;
  p.n_components = 2;
  p.random_state = 5;
  PCA pca(p);
  Features projected = pca.fit_transform(X);

  float ratio_sum = pca.explained_variance_ratio()[0] + pca.explained_variance_ratio()[1];

  double total_sq_error = 0.0, total_sq_signal = 0.0;
  for (std::size_t i = 0; i < X.size(); ++i) {
    std::vector<float> recon(5, 0.0f);
    for (std::size_t f = 0; f < 5; ++f) recon[f] = pca.components()[0][f] * projected[i][0] + pca.components()[1][f] * projected[i][1];
    // recon is relative to the mean (already ~0 for centered Gaussian data); compare against mean-centered X.
    for (std::size_t f = 0; f < 5; ++f) {
      float diff = X[i][f] - recon[f];
      total_sq_error += static_cast<double>(diff) * static_cast<double>(diff);
      total_sq_signal += static_cast<double>(X[i][f]) * static_cast<double>(X[i][f]);
    }
  }
  double relative_error = total_sq_error / total_sq_signal;

  std::printf("  top-2 explained_variance_ratio sum=%.4f, reconstruction relative error=%.4f\n", static_cast<double>(ratio_sum),
              relative_error);
  require(ratio_sum > 0.98f, "top 2 components capture >98% of variance on data that is genuinely 2D plus small noise");
  require(relative_error < 0.02, "reconstructing from just the top 2 components recovers the original 5D data almost exactly");
}

// Fundamental SVD invariant: principal components must be unit-norm and
// mutually orthogonal, regardless of the data.
void test_components_are_orthonormal() {
  std::mt19937 rng(2);
  std::normal_distribution<float> dist(0.0f, 2.0f);
  Features X;
  for (int i = 0; i < 100; ++i) X.push_back({dist(rng), dist(rng), dist(rng), dist(rng)});

  PCAParams p;
  p.n_components = 4;
  p.random_state = 9;
  PCA pca(p);
  pca.fit(X);

  bool all_unit_norm = true, all_orthogonal = true;
  for (std::size_t i = 0; i < pca.components().size(); ++i) {
    float norm = std::sqrt(dot(pca.components()[i], pca.components()[i]));
    if (std::fabs(norm - 1.0f) > 1e-3f) all_unit_norm = false;
    for (std::size_t j = i + 1; j < pca.components().size(); ++j) {
      if (std::fabs(dot(pca.components()[i], pca.components()[j])) > 1e-3f) all_orthogonal = false;
    }
  }

  std::printf("  components: all_unit_norm=%d all_orthogonal=%d\n", all_unit_norm, all_orthogonal);
  require(all_unit_norm, "every principal component has unit norm");
  require(all_orthogonal, "every pair of principal components is orthogonal");
}

// At full rank (n_components == n_features), explained variance ratios
// must sum to ~1 -- all of the data's variance is accounted for once
// nothing is truncated.
void test_full_rank_explained_variance_sums_to_one() {
  std::mt19937 rng(3);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  Features X;
  for (int i = 0; i < 150; ++i) X.push_back({dist(rng), dist(rng) * 2.0f, dist(rng) * 0.5f});

  PCAParams p;
  p.n_components = 3;
  p.random_state = 1;
  PCA pca(p);
  pca.fit(X);

  float sum = 0.0f;
  for (float r : pca.explained_variance_ratio()) sum += r;
  bool descending = pca.explained_variance_ratio()[0] >= pca.explained_variance_ratio()[1] &&
                     pca.explained_variance_ratio()[1] >= pca.explained_variance_ratio()[2];

  std::printf("  full-rank explained_variance_ratio sum=%.4f, descending=%d\n", static_cast<double>(sum), descending);
  require(std::fabs(sum - 1.0f) < 1e-2f, "at full rank, explained variance ratios sum to ~1 (all variance accounted for)");
  require(descending, "explained variance ratios are sorted descending by component");
}

// 2D data with variance 10x larger along one axis than the other: the
// first principal component should align with the high-variance axis
// (up to an overall sign flip, which SVD leaves undetermined), and it
// should capture the large majority of the variance.
void test_first_component_aligns_with_dominant_variance_axis() {
  std::mt19937 rng(4);
  std::normal_distribution<float> wide(0.0f, 10.0f), narrow(0.0f, 1.0f);
  Features X;
  for (int i = 0; i < 200; ++i) X.push_back({wide(rng), narrow(rng)});

  PCAParams p;
  p.n_components = 2;
  p.random_state = 2;
  PCA pca(p);
  pca.fit(X);

  float alignment = std::fabs(pca.components()[0][0]);  // |dot with unit x-axis vector [1,0]|

  std::printf("  first component=[%.3f, %.3f], explained_variance_ratio[0]=%.4f\n", static_cast<double>(pca.components()[0][0]),
              static_cast<double>(pca.components()[0][1]), static_cast<double>(pca.explained_variance_ratio()[0]));
  require(alignment > 0.99f, "the first principal component aligns with the axis carrying 10x more variance");
  require(pca.explained_variance_ratio()[0] > 0.95f, "the dominant axis accounts for >95% of variance");
}

// Whitening's whole point: transformed features should have ~unit
// variance per component, undoing the very variance differences the
// unwhitened test above measures.
void test_whitening_produces_unit_variance_components() {
  std::mt19937 rng(5);
  std::normal_distribution<float> wide(0.0f, 10.0f), narrow(0.0f, 1.0f);
  Features X;
  for (int i = 0; i < 300; ++i) X.push_back({wide(rng), narrow(rng)});

  PCAParams p;
  p.n_components = 2;
  p.whiten = true;
  p.random_state = 3;
  PCA pca(p);
  Features transformed = pca.fit_transform(X);

  for (int c = 0; c < 2; ++c) {
    double mean = 0.0;
    for (const auto &row : transformed) mean += static_cast<double>(row[static_cast<std::size_t>(c)]);
    mean /= static_cast<double>(transformed.size());
    double var = 0.0;
    for (const auto &row : transformed) {
      double diff = static_cast<double>(row[static_cast<std::size_t>(c)]) - mean;
      var += diff * diff;
    }
    var /= (transformed.size() - 1);
    std::printf("  whitened component %d variance=%.4f\n", c, var);
    require(std::fabs(var - 1.0) < 0.15, "whitened component variance is close to 1");
  }
}

}  // namespace

int main() {
  test_pca_reconstructs_low_rank_embedded_data();
  test_components_are_orthonormal();
  test_full_rank_explained_variance_sums_to_one();
  test_first_component_aligns_with_dominant_variance_axis();
  test_whitening_produces_unit_variance_components();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
