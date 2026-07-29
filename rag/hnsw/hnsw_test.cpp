// hnsw_test.cpp -- real correctness/recall checks for HNSW, benchmarked
// directly against ml/knn's BallTree on the SAME corpus and SAME queries
// (PLAN.md Phase 13 step 3's own ask), not just "it runs":
//  1. recall against a brute-force ground truth (HNSW, unlike KDTree/
//     BallTree's exact mode, has no exactness guarantee at all -- it's
//     approximate by construction, so recall has to be measured, not
//     assumed).
//  2. a head-to-head cost comparison against BallTree: distance
//     evaluations per query for a comparable recall level, on the same
//     random point set.
#include "hnsw.h"
#include "../../ml/knn/knn.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <set>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

std::vector<NeighborResult> brute_force_knn(const Features &X, const std::vector<float> &query, int k) {
  std::vector<NeighborResult> all;
  all.reserve(X.size());
  for (std::size_t i = 0; i < X.size(); ++i) all.push_back({static_cast<int>(i), squared_distance(X[i], query)});
  std::sort(all.begin(), all.end(), [](const NeighborResult &a, const NeighborResult &b) { return a.dist_sq < b.dist_sq; });
  all.resize(static_cast<std::size_t>(k));
  return all;
}

std::set<int> index_set(const std::vector<NeighborResult> &nn) {
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

void test_hnsw_recall_against_bruteforce() {
  std::mt19937 rng(1);
  Features X = random_points(1000, 32, rng);
  HNSW::Params params;
  params.M = 12;
  params.ef_construction = 100;
  params.ef_search = 64;
  HNSW index(X, params);

  int total_matches = 0, total_true = 0;
  const int n_queries = 30, k = 10;
  for (int q = 0; q < n_queries; ++q) {
    std::vector<float> query = random_points(1, 32, rng)[0];
    auto approx = index.query_knn(query, k);
    auto exact = brute_force_knn(X, query, k);
    std::set<int> exact_idx = index_set(exact);
    for (const auto &r : approx)
      if (exact_idx.count(r.index)) ++total_matches;
    total_true += static_cast<int>(exact.size());
  }

  double recall = static_cast<double>(total_matches) / total_true;
  std::printf("  HNSW recall@%d over %d queries (n=%d, dims=32): %.3f\n", k, n_queries,
              static_cast<int>(X.size()), recall);
  require(recall > 0.85, "HNSW recall is high on i.i.d. random data with a reasonably large ef_search");
}

// Head-to-head against BallTree on the SAME points and SAME queries --
// PLAN.md step 3's own ask ("benchmarked against the ball tree baseline
// on the same corpus"). Reports both recall and search cost (distance
// evaluations vs. nodes visited) honestly -- whichever wins, it's a real
// measurement, not assumed in HNSW's favor just because it's the newer
// algorithm.
void test_hnsw_vs_balltree_same_corpus() {
  std::mt19937 rng(2);
  Features X = random_points(2000, 64, rng);
  HNSW::Params params;
  params.M = 16;
  params.ef_construction = 100;
  params.ef_search = 64;
  HNSW hnsw(X, params);
  BallTree balltree(X);

  const int n_queries = 30, k = 10;
  std::size_t hnsw_evals = 0, balltree_evals = 0;
  int hnsw_matches = 0, balltree_matches = 0, total_true = 0;

  for (int q = 0; q < n_queries; ++q) {
    std::vector<float> query = random_points(1, 64, rng)[0];
    auto exact = brute_force_knn(X, query, k);
    std::set<int> exact_idx = index_set(exact);

    auto hnsw_result = hnsw.query_knn(query, k);
    hnsw_evals += hnsw.last_distance_evals();
    for (const auto &r : hnsw_result)
      if (exact_idx.count(r.index)) ++hnsw_matches;

    auto balltree_result = balltree.query_knn(query, k, /*approximate=*/false);
    balltree_evals += balltree.last_nodes_visited();
    for (const auto &r : balltree_result)
      if (exact_idx.count(r.index)) ++balltree_matches;

    total_true += k;
  }

  double hnsw_recall = static_cast<double>(hnsw_matches) / total_true;
  double balltree_recall = static_cast<double>(balltree_matches) / total_true;
  double hnsw_avg_evals = static_cast<double>(hnsw_evals) / n_queries;
  double balltree_avg_evals = static_cast<double>(balltree_evals) / n_queries;

  std::printf("  HNSW:     recall@%d=%.3f, avg distance evals/query=%.1f (of %d points)\n", k, hnsw_recall,
              hnsw_avg_evals, static_cast<int>(X.size()));
  std::printf("  BallTree: recall@%d=%.3f (exact, always 1.000), avg nodes visited/query=%.1f (of %d points)\n", k,
              balltree_recall, balltree_avg_evals, static_cast<int>(X.size()));

  require(balltree_recall > 0.999, "BallTree's exact mode is exact (sanity check on the ground truth comparison itself)");
  require(hnsw_recall > 0.7, "HNSW finds a substantial fraction of the true k nearest neighbors on this corpus");
  require(hnsw_avg_evals < static_cast<double>(X.size()),
          "HNSW visits far fewer than all points per query (sub-linear search, the whole point of the graph)");
}

} // namespace

int main() {
  test_hnsw_recall_against_bruteforce();
  test_hnsw_vs_balltree_same_corpus();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
