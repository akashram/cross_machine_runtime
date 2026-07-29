// cosine_ann_test.cpp -- real correctness checks for CosineBallTree, same
// spirit as knn_test.cpp: verified against a brute-force cosine-similarity
// ground truth, not just "it runs". Also verifies the core claim
// cosine_ann.h's design note makes directly -- that ranking by Euclidean
// distance on L2-normalized vectors is EXACTLY equivalent to ranking by
// cosine similarity -- rather than just asserting it.
#include "cosine_ann.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

std::vector<CosineNeighborResult> brute_force_cosine_knn(const Features &X, const std::vector<float> &query, int k) {
  std::vector<CosineNeighborResult> all;
  all.reserve(X.size());
  for (std::size_t i = 0; i < X.size(); ++i) all.push_back({static_cast<int>(i), cosine_similarity(X[i], query)});
  std::sort(all.begin(), all.end(),
            [](const CosineNeighborResult &a, const CosineNeighborResult &b) { return a.similarity > b.similarity; });
  all.resize(static_cast<std::size_t>(k));
  return all;
}

std::set<int> index_set(const std::vector<CosineNeighborResult> &nn) {
  std::set<int> s;
  for (const auto &n : nn) s.insert(n.index);
  return s;
}

Features random_points(int n, int dims, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  Features X;
  X.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    std::vector<float> row(static_cast<std::size_t>(dims));
    for (int d = 0; d < dims; ++d) row[static_cast<std::size_t>(d)] = dist(rng);
    X.push_back(std::move(row));
  }
  return X;
}

// Textbook sanity checks on cosine_similarity() itself before trusting it
// as ground truth for anything else.
void test_cosine_similarity_basic_cases() {
  bool ok_parallel = std::abs(cosine_similarity({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 6.0f}) - 1.0f) < 1e-5f;
  require(ok_parallel, "cosine_similarity of parallel (same-direction) vectors is 1.0");

  bool ok_orthogonal = std::abs(cosine_similarity({1.0f, 0.0f}, {0.0f, 1.0f})) < 1e-6f;
  require(ok_orthogonal, "cosine_similarity of orthogonal vectors is 0.0");

  bool ok_opposite = std::abs(cosine_similarity({1.0f, 2.0f}, {-1.0f, -2.0f}) - (-1.0f)) < 1e-5f;
  require(ok_opposite, "cosine_similarity of opposite-direction vectors is -1.0");
}

// The core claim: CosineBallTree's exact-mode ranking (Euclidean search on
// normalized vectors) must match a brute-force cosine-similarity ranking
// exactly -- the monotonic-transform argument in cosine_ann.h, verified
// directly rather than assumed.
void test_cosine_balltree_matches_bruteforce_cosine() {
  std::mt19937 rng(1);
  Features X = random_points(300, 8, rng);
  CosineBallTree tree(X);

  int mismatches = 0;
  int similarity_mismatches = 0;
  const int n_queries = 20;
  for (int q = 0; q < n_queries; ++q) {
    std::vector<float> query = random_points(1, 8, rng)[0];
    auto exact = tree.query_knn(query, 5, /*approximate=*/false);
    auto brute = brute_force_cosine_knn(X, query, 5);
    if (index_set(exact) != index_set(brute)) ++mismatches;

    // The similarity value CosineBallTree reports (derived from the
    // normalized-space squared distance) should match direct
    // cosine_similarity() on the ORIGINAL (unnormalized) vectors.
    for (const auto &r : exact) {
      float direct = cosine_similarity(X[static_cast<std::size_t>(r.index)], query);
      if (std::abs(direct - r.similarity) > 1e-4f) ++similarity_mismatches;
    }
  }

  std::printf("  CosineBallTree: %d/%d queries mismatched brute-force cosine ranking; %d/%d similarity values off\n",
              mismatches, n_queries, similarity_mismatches, n_queries * 5);
  require(mismatches == 0, "CosineBallTree's exact mode finds the same top-k as brute-force cosine similarity");
  require(similarity_mismatches == 0, "CosineBallTree reports the correct cosine similarity value, not just the correct order");
}

// Same approximate/exact recall-vs-speed measurement knn_test.cpp makes
// for plain BallTree, carried over to the cosine wrapper: defeatist search
// should visit fewer nodes but not be a free lunch (real recall loss).
void test_cosine_balltree_approximate_tradeoff() {
  std::mt19937 rng(3);
  const int n_clusters = 5, per_cluster = 60, dims = 12;
  Features X;
  std::vector<int> cluster_of;
  std::normal_distribution<float> spread(0.0f, 1.0f);
  for (int c = 0; c < n_clusters; ++c) {
    std::vector<float> center(static_cast<std::size_t>(dims));
    std::uniform_real_distribution<float> center_dist(-20.0f, 20.0f);
    for (float &v : center) v = center_dist(rng);
    for (int i = 0; i < per_cluster; ++i) {
      std::vector<float> pt(static_cast<std::size_t>(dims));
      for (int d = 0; d < dims; ++d) pt[static_cast<std::size_t>(d)] = center[static_cast<std::size_t>(d)] + spread(rng);
      X.push_back(pt);
      cluster_of.push_back(c);
    }
  }
  CosineBallTree tree(X, /*leaf_size=*/20);

  int total_matches = 0, total_true = 0;
  std::size_t visited_exact = 0, visited_approx = 0;
  const int n_queries = 30;
  for (int q = 0; q < n_queries; ++q) {
    int c = static_cast<int>(rng() % static_cast<unsigned>(n_clusters));
    std::vector<float> query = X[static_cast<std::size_t>(c * per_cluster)];
    for (float &v : query) v += spread(rng) * 0.1f;

    auto exact = tree.query_knn(query, 5, /*approximate=*/false);
    visited_exact += tree.last_nodes_visited();
    auto approx = tree.query_knn(query, 5, /*approximate=*/true);
    visited_approx += tree.last_nodes_visited();

    std::set<int> exact_idx = index_set(exact);
    for (const auto &r : approx)
      if (exact_idx.count(r.index)) ++total_matches;
    total_true += static_cast<int>(exact.size());
  }

  double recall = static_cast<double>(total_matches) / total_true;
  std::printf("  CosineBallTree defeatist: recall=%.3f vs exact, avg nodes visited exact=%.1f approx=%.1f\n", recall,
              static_cast<double>(visited_exact) / n_queries, static_cast<double>(visited_approx) / n_queries);
  require(recall < 1.0, "defeatist cosine search actually misses some true neighbors (a genuine approximation)");
  require(recall > 0.3, "defeatist cosine search still finds a nontrivial fraction of true neighbors");
  require(visited_approx < visited_exact, "defeatist cosine search visits fewer nodes than exact backtracking search");
}

}  // namespace

int main() {
  test_cosine_similarity_basic_cases();
  test_cosine_balltree_matches_bruteforce_cosine();
  test_cosine_balltree_approximate_tradeoff();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
