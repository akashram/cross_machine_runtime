// knn_test.cpp — real correctness checks, not just "it runs": both
// KDTree exact and BallTree exact are checked against a brute-force O(n)
// ground truth (branch-and-bound pruning must never change the answer),
// BallTree's approximate (defeatist) mode is shown to actually trade
// recall for fewer node visits rather than silently being exact, and
// KNNClassifier demonstrates the classic k bias-variance tradeoff on
// noisy labels (same spirit as svm_test.cpp's C sweep, decision_tree's
// depth sweep).
#include "knn.h"

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
  for (std::size_t i = 0; i < X.size(); ++i)
    all.push_back({static_cast<int>(i), squared_distance(X[i], query)});
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

// Branch-and-bound pruning must never change the answer: KDTree's
// result set has to match brute force exactly, on every query.
void test_kdtree_matches_bruteforce() {
  std::mt19937 rng(1);
  Features X = random_points(300, 5, rng);
  KDTree tree(X);

  int mismatches = 0;
  std::size_t total_visited = 0;
  const int n_queries = 20;
  for (int q = 0; q < n_queries; ++q) {
    std::vector<float> query = random_points(1, 5, rng)[0];
    auto exact = tree.query_knn(query, 5);
    auto brute = brute_force_knn(X, query, 5);
    if (index_set(exact) != index_set(brute)) ++mismatches;
    total_visited += tree.last_nodes_visited();
  }

  std::printf("  KDTree: %d/%d queries mismatched brute force; avg nodes visited=%.1f / %d points\n", mismatches,
              n_queries, static_cast<double>(total_visited) / n_queries, static_cast<int>(X.size()));
  require(mismatches == 0, "KDTree's branch-and-bound pruning finds the same k nearest neighbors as brute force");
  require(static_cast<double>(total_visited) / n_queries < static_cast<double>(X.size()),
          "KDTree visits fewer nodes than a full brute-force scan (pruning actually prunes)");
}

// Same exactness guarantee for BallTree's non-approximate mode -- the
// triangle-inequality radius bound must be a real bound, not a heuristic.
void test_balltree_exact_matches_bruteforce() {
  std::mt19937 rng(2);
  Features X = random_points(300, 5, rng);
  BallTree tree(X);

  int mismatches = 0;
  const int n_queries = 20;
  for (int q = 0; q < n_queries; ++q) {
    std::vector<float> query = random_points(1, 5, rng)[0];
    auto exact = tree.query_knn(query, 5, /*approximate=*/false);
    auto brute = brute_force_knn(X, query, 5);
    if (index_set(exact) != index_set(brute)) ++mismatches;
  }

  std::printf("  BallTree (exact): %d/%d queries mismatched brute force\n", mismatches, n_queries);
  require(mismatches == 0, "BallTree's exact (bound-and-backtrack) mode finds the same k nearest neighbors as brute force");
}

// The whole point of PLAN.md's "ball tree (approximate)": defeatist
// search (no backtracking) should visit noticeably fewer nodes than
// exact search, and -- because it never checks the sibling subtree --
// it should actually miss some true neighbors sometimes, not
// coincidentally always match. If this ever measures 100% recall, the
// test setup isn't actually exercising approximation.
void test_balltree_defeatist_trades_recall_for_speed() {
  std::mt19937 rng(3);
  // Higher dimensionality and several overlapping clusters: the regime
  // where a single defeatist descent is more likely to land in the
  // "wrong" nearby leaf and miss neighbors just across a ball boundary.
  Features X;
  std::vector<std::vector<float>> centers;
  std::normal_distribution<float> noise(0.0f, 3.0f);
  std::uniform_real_distribution<float> center_dist(-15.0f, 15.0f);
  for (int c = 0; c < 5; ++c) {
    std::vector<float> center(16);
    for (auto &v : center) v = center_dist(rng);
    centers.push_back(center);
    for (int i = 0; i < 60; ++i) {
      std::vector<float> row(16);
      for (int d = 0; d < 16; ++d) row[static_cast<std::size_t>(d)] = center[static_cast<std::size_t>(d)] + noise(rng);
      X.push_back(std::move(row));
    }
  }
  BallTree tree(X, /*leaf_size=*/20);

  // Queries drawn in-distribution (near a random cluster center, same
  // noise as training data) -- the realistic kNN usage pattern of a new
  // sample from the same distribution as training data, not an
  // arbitrary point in empty space where "nearest neighbor" is close to
  // meaningless.
  std::uniform_int_distribution<int> pick_center(0, static_cast<int>(centers.size()) - 1);
  int k = 5;
  int n_queries = 40;
  int total_found = 0;
  std::size_t exact_visited = 0, approx_visited = 0;
  for (int q = 0; q < n_queries; ++q) {
    const auto &center = centers[static_cast<std::size_t>(pick_center(rng))];
    std::vector<float> query(16);
    for (int d = 0; d < 16; ++d) query[static_cast<std::size_t>(d)] = center[static_cast<std::size_t>(d)] + noise(rng);
    auto exact = tree.query_knn(query, k, false);
    exact_visited += tree.last_nodes_visited();
    auto approx = tree.query_knn(query, k, true);
    approx_visited += tree.last_nodes_visited();

    std::set<int> exact_set = index_set(exact);
    for (const auto &n : approx)
      if (exact_set.count(n.index)) ++total_found;
  }
  double recall = static_cast<double>(total_found) / (n_queries * k);

  std::printf("  BallTree defeatist: recall=%.3f vs exact, avg nodes visited exact=%.1f approx=%.1f\n", recall,
              static_cast<double>(exact_visited) / n_queries, static_cast<double>(approx_visited) / n_queries);
  require(recall < 1.0, "defeatist search actually misses some true neighbors on this data (a genuine approximation, not accidentally exact)");
  require(recall > 0.25, "defeatist search still finds a nontrivial fraction of true neighbors (not landing in an unrelated cluster)");
  require(approx_visited < exact_visited, "defeatist search visits fewer nodes than exact backtracking search (the speed side of the tradeoff)");
}

// Classic k bias-variance tradeoff: with a fraction of training labels
// flipped near the boundary, a large k should out-vote individual noisy
// neighbors more often than k=1, which just copies whichever single
// point happens to be nearest (noisy or not).
void test_larger_k_more_robust_to_label_noise() {
  std::mt19937 rng(4);
  std::normal_distribution<float> spread(0.0f, 1.0f);
  std::uniform_real_distribution<float> flip_roll(0.0f, 1.0f);
  Features X_train;
  Labels y_train;
  for (int i = 0; i < 150; ++i) {
    X_train.push_back({-3.0f + spread(rng), -3.0f + spread(rng)});
    y_train.push_back(flip_roll(rng) < 0.25f ? 1.0f : 0.0f);
  }
  for (int i = 0; i < 150; ++i) {
    X_train.push_back({3.0f + spread(rng), 3.0f + spread(rng)});
    y_train.push_back(flip_roll(rng) < 0.25f ? 0.0f : 1.0f);
  }

  // Clean held-out test set (true labels, no noise) -- what actually
  // demonstrates generalization rather than memorized noise.
  Features X_test;
  Labels y_test;
  for (int i = 0; i < 60; ++i) {
    X_test.push_back({-3.0f + spread(rng), -3.0f + spread(rng)});
    y_test.push_back(0.0f);
  }
  for (int i = 0; i < 60; ++i) {
    X_test.push_back({3.0f + spread(rng), 3.0f + spread(rng)});
    y_test.push_back(1.0f);
  }

  KNNParams p1;
  p1.k = 1;
  KNNClassifier knn1(p1);
  knn1.fit(X_train, y_train);
  float acc_k1 = knn1.score(X_test, y_test);

  KNNParams p15;
  p15.k = 15;
  KNNClassifier knn15(p15);
  knn15.fit(X_train, y_train);
  float acc_k15 = knn15.score(X_test, y_test);

  std::printf("  held-out accuracy: k=1 -> %.3f, k=15 -> %.3f\n", static_cast<double>(acc_k1), static_cast<double>(acc_k15));
  require(acc_k15 >= acc_k1, "a larger k generalizes at least as well as k=1 on noisy-label training data (majority vote out-votes individual noisy neighbors)");
}

// Sanity check that KDTree and BallTree (exact) agree on the same
// classification task -- two different tree structures over the same
// exact-neighbor semantics should never disagree.
void test_kdtree_and_balltree_agree_on_classification() {
  std::mt19937 rng(5);
  std::normal_distribution<float> noise(0.0f, 0.5f);
  Features X;
  Labels y;
  for (int i = 0; i < 100; ++i) {
    X.push_back({-2.0f + noise(rng), -2.0f + noise(rng)});
    y.push_back(0.0f);
  }
  for (int i = 0; i < 100; ++i) {
    X.push_back({2.0f + noise(rng), 2.0f + noise(rng)});
    y.push_back(1.0f);
  }

  KNNParams kd_p;
  kd_p.structure = NeighborStructure::KD_TREE;
  kd_p.k = 3;
  KNNClassifier kd_knn(kd_p);
  kd_knn.fit(X, y);

  KNNParams ball_p;
  ball_p.structure = NeighborStructure::BALL_TREE;
  ball_p.approximate = false;
  ball_p.k = 3;
  KNNClassifier ball_knn(ball_p);
  ball_knn.fit(X, y);

  float kd_acc = kd_knn.score(X, y);
  float ball_acc = ball_knn.score(X, y);

  std::printf("  well-separated blobs train accuracy: KD_TREE=%.3f BALL_TREE(exact)=%.3f\n", static_cast<double>(kd_acc),
              static_cast<double>(ball_acc));
  require(kd_acc == ball_acc, "KDTree and exact BallTree produce identical classification accuracy (same exact-neighbor semantics)");
  require(kd_acc > 0.98f, "k-NN separates two well-separated blobs almost exactly");
}

}  // namespace

int main() {
  test_kdtree_matches_bruteforce();
  test_balltree_exact_matches_bruteforce();
  test_balltree_defeatist_trades_recall_for_speed();
  test_larger_k_more_robust_to_label_noise();
  test_kdtree_and_balltree_agree_on_classification();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
