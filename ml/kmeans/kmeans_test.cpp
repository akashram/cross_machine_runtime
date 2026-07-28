// kmeans_test.cpp — real correctness checks: cluster assignments
// actually recover the ground-truth blob structure, the elbow curve
// actually elbows at the true k, and k-means++ is measured (not
// asserted) to beat plain random init on a scenario prone to the bad
// local optima random init is known to hit.
#include "kmeans.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <set>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

// Fraction of points whose predicted cluster label agrees with the
// majority predicted label within their true blob -- a simple purity
// measure that doesn't require label identity to match blob identity
// (k-means labels are arbitrary permutations of 0..k-1).
float clustering_purity(const std::vector<int> &predicted, const std::vector<int> &true_blob, int n_blobs) {
  std::vector<std::map<int, int>> votes(static_cast<std::size_t>(n_blobs));
  for (std::size_t i = 0; i < predicted.size(); ++i) ++votes[static_cast<std::size_t>(true_blob[i])][predicted[i]];
  int correct = 0;
  for (int b = 0; b < n_blobs; ++b) {
    int best = 0;
    for (auto &[label, count] : votes[static_cast<std::size_t>(b)]) best = std::max(best, count);
    correct += best;
  }
  return static_cast<float>(correct) / static_cast<float>(predicted.size());
}

// Three well-separated 2D Gaussian blobs: k-means with k=3 should
// recover the blob structure almost exactly.
void test_kmeans_recovers_well_separated_blobs() {
  std::mt19937 rng(1);
  std::normal_distribution<float> noise(0.0f, 0.4f);
  Features X;
  std::vector<int> true_blob;
  std::vector<std::vector<float>> centers = {{-5.0f, -5.0f}, {5.0f, -5.0f}, {0.0f, 5.0f}};
  for (int b = 0; b < 3; ++b) {
    for (int i = 0; i < 100; ++i) {
      X.push_back({centers[static_cast<std::size_t>(b)][0] + noise(rng), centers[static_cast<std::size_t>(b)][1] + noise(rng)});
      true_blob.push_back(b);
    }
  }

  KMeansParams p;
  p.k = 3;
  p.random_state = 42;
  KMeans km(p);
  km.fit(X);
  float purity = clustering_purity(km.labels(), true_blob, 3);

  std::printf("  3-blob clustering purity=%.3f, converged in %d iterations, inertia=%.2f\n", static_cast<double>(purity),
              km.n_iter(), static_cast<double>(km.inertia()));
  require(purity > 0.98f, "k-means with k=3 recovers three well-separated blobs almost exactly");
}

// The elbow method's whole premise: inertia should drop sharply up to
// the true k, then flatten out -- measured here as the marginal drop
// from k=2->3 (still splitting real clusters) being much larger than
// from k=5->6 (splitting an already-correct cluster in half, which
// barely helps).
void test_elbow_curve_flattens_past_true_k() {
  std::mt19937 rng(2);
  std::normal_distribution<float> noise(0.0f, 0.4f);
  Features X;
  std::vector<std::vector<float>> centers = {{-5.0f, -5.0f}, {5.0f, -5.0f}, {0.0f, 5.0f}};
  for (const auto &c : centers)
    for (int i = 0; i < 100; ++i) X.push_back({c[0] + noise(rng), c[1] + noise(rng)});

  std::vector<float> inertias = elbow_curve(X, 6, 7);
  float drop_2_to_3 = inertias[1] - inertias[2];
  float drop_5_to_6 = inertias[4] - inertias[5];

  std::printf("  elbow inertia by k: ");
  for (std::size_t i = 0; i < inertias.size(); ++i) std::printf("k=%zu:%.1f ", i + 1, static_cast<double>(inertias[i]));
  std::printf("\n  drop(2->3)=%.2f drop(5->6)=%.2f\n", static_cast<double>(drop_2_to_3), static_cast<double>(drop_5_to_6));
  require(drop_2_to_3 > 5.0f * drop_5_to_6,
          "inertia drops far more sharply going from k=2 to the true k=3 than from k=5 to k=6 past it (the elbow)");
}

// k-means++'s actual point, measured against a real random-init
// baseline (not assumed): on many small, similarly-sized clusters --
// where plain random init is likely to place two initial centroids in
// the same cluster and starve another -- k-means++ should reach lower
// average inertia and a much lower worst-case (max) inertia across
// random seeds, since it structurally can't place two centroids right
// next to each other the way independent uniform sampling can.
void test_kmeanspp_beats_random_init_on_many_clusters() {
  std::mt19937 data_rng(3);
  std::normal_distribution<float> noise(0.0f, 0.3f);
  std::uniform_real_distribution<float> center_dist(-20.0f, 20.0f);
  Features X;
  int n_clusters = 8;
  for (int c = 0; c < n_clusters; ++c) {
    float cx = center_dist(data_rng), cy = center_dist(data_rng);
    for (int i = 0; i < 20; ++i) X.push_back({cx + noise(data_rng), cy + noise(data_rng)});
  }

  int n_seeds = 25;
  float sum_pp = 0.0f, sum_random = 0.0f, max_pp = 0.0f, max_random = 0.0f;
  for (unsigned seed = 0; seed < static_cast<unsigned>(n_seeds); ++seed) {
    KMeansParams pp_params;
    pp_params.k = n_clusters;
    pp_params.random_state = seed;
    pp_params.use_kmeanspp_init = true;
    KMeans pp(pp_params);
    pp.fit(X);
    sum_pp += pp.inertia();
    max_pp = std::max(max_pp, pp.inertia());

    KMeansParams random_params = pp_params;
    random_params.use_kmeanspp_init = false;
    KMeans random_km(random_params);
    random_km.fit(X);
    sum_random += random_km.inertia();
    max_random = std::max(max_random, random_km.inertia());
  }

  float avg_pp = sum_pp / static_cast<float>(n_seeds);
  float avg_random = sum_random / static_cast<float>(n_seeds);
  std::printf("  over %d seeds: kmeans++ avg inertia=%.2f (worst=%.2f), random-init avg inertia=%.2f (worst=%.2f)\n", n_seeds,
              static_cast<double>(avg_pp), static_cast<double>(max_pp), static_cast<double>(avg_random), static_cast<double>(max_random));
  require(avg_pp <= avg_random, "k-means++ reaches lower (or equal) average inertia than plain random init across seeds");
  require(max_pp < max_random, "k-means++'s worst-case seed is still better than random init's worst-case seed (fewer bad local optima)");
}

// predict() on held-out points must use the fitted centroids, not
// refit -- points near a training blob's center should be assigned to
// that blob's label.
void test_predict_assigns_new_points_to_nearest_fitted_centroid() {
  std::mt19937 rng(4);
  std::normal_distribution<float> noise(0.0f, 0.3f);
  Features X;
  std::vector<std::vector<float>> centers = {{-5.0f, -5.0f}, {5.0f, -5.0f}, {0.0f, 5.0f}};
  for (const auto &c : centers)
    for (int i = 0; i < 60; ++i) X.push_back({c[0] + noise(rng), c[1] + noise(rng)});

  KMeansParams p;
  p.k = 3;
  p.random_state = 11;
  KMeans km(p);
  km.fit(X);

  // Held-out points, one very close to each known blob center.
  Features held_out = {{-5.1f, -4.9f}, {4.9f, -5.1f}, {0.1f, 5.1f}};
  std::vector<int> pred = km.predict(held_out);
  bool all_distinct = std::set<int>(pred.begin(), pred.end()).size() == 3;

  std::printf("  held-out predictions: %d %d %d\n", pred[0], pred[1], pred[2]);
  require(all_distinct, "three held-out points near three different blob centers get three different predicted labels");
}

}  // namespace

int main() {
  test_kmeans_recovers_well_separated_blobs();
  test_elbow_curve_flattens_past_true_k();
  test_kmeanspp_beats_random_init_on_many_clusters();
  test_predict_assigns_new_points_to_nearest_fitted_centroid();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
