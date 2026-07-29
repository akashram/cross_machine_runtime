#pragma once
#include "../../ml/knn/knn.h" // Features, squared_distance, NeighborResult

#include <cstddef>
#include <random>
#include <vector>

// PLAN.md Phase 13 step 3: HNSW (Hierarchical Navigable Small World graphs,
// Malkov & Yashunin 2016) -- a more scalable ANN structure than
// ml/knn's BallTree, benchmarked against it directly on the same corpus
// (see hnsw_test.cpp). Reuses ml/knn's Features/NeighborResult/
// squared_distance rather than redefining them.
//
// Simplified vs. the full paper (same "real but simplified, honestly
// documented" convention as ml/knn's BallTree/KDTree): neighbor selection
// during insertion uses the SIMPLE heuristic (keep the M closest
// candidates) rather than the paper's more elaborate diversification
// heuristic (Algorithm 4, which also tries to keep neighbors spread
// across directions, not just close) -- the simple heuristic still
// produces a real, correct, functioning small-world graph, just with
// somewhat lower recall/build-quality than the full heuristic would.
//
// Structure: each point gets a random top layer (exponentially decaying,
// Malkov & Yashunin's mL = 1/ln(M) parameterization), building a
// hierarchy where higher layers hold exponentially fewer nodes -- entry
// at the top layer greedily narrows down to the right neighborhood before
// layer 0's exhaustive-within-ef search takes over, which is what makes
// search O(log n)-ish instead of visiting a large fraction of the graph.
class HNSW {
public:
  struct Params {
    int M = 8;              // neighbors per node per layer (2*M at layer 0)
    int ef_construction = 64;
    int ef_search = 32;
    unsigned seed = 42;
  };

  // Note: the default-argument form `Params params = {}` triggers a real
  // clang parsing restriction (a nested class's default member
  // initializers aren't considered available until the OUTERMOST
  // enclosing class is complete, even inside an inline mem-initializer
  // list), so the default is provided via a delegating constructor
  // defined out-of-line in hnsw.cpp, after HNSW's closing brace.
  explicit HNSW(const Features &points);
  explicit HNSW(const Features &points, Params params);

  std::vector<NeighborResult> query_knn(const std::vector<float> &query, int k) const;

  // Distance evaluations made by the most recent query_knn() call -- the
  // HNSW analogue of BallTree/KDTree's last_nodes_visited(), letting
  // hnsw_test.cpp compare search cost directly against BallTree's.
  std::size_t last_distance_evals() const { return last_distance_evals_; }

private:
  struct Node {
    std::vector<std::vector<int>> layers; // neighbor ids per layer, layers[0] = layer 0
  };

  Features points_;
  Params params_;
  std::vector<Node> nodes_;
  int entry_point_ = -1;
  int max_layer_ = -1;
  mutable std::size_t last_distance_evals_ = 0;
  std::mt19937 rng_;
  float mL_;

  int random_level();
  void insert(int idx);
  // Algorithm 2 (SEARCH-LAYER) from Malkov & Yashunin 2016: greedy
  // best-first graph exploration bounded by `ef` candidates, within a
  // single layer.
  std::vector<NeighborResult> search_layer(const std::vector<float> &query, const std::vector<int> &entry_points,
                                            int ef, int layer) const;
  static std::vector<int> select_neighbors_simple(std::vector<NeighborResult> candidates, int m);
};
