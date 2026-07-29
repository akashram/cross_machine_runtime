#pragma once
#include "knn.h"

// PLAN.md Phase 13 step 2: cosine-similarity ANN retrieval, built as a
// thin wrapper AROUND ml/knn's existing BallTree rather than a modified
// copy of it -- composition, not duplication (same "RAG is a composition
// of existing capabilities" note PLAN.md's Phase 13 section makes).
//
// The trick this relies on (a standard one, verified directly here rather
// than just asserted -- see cosine_ann_test.cpp): for UNIT vectors,
// squared Euclidean distance and cosine similarity are a monotonic
// transform of each other --
//   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b = 2 - 2*cos(a, b)   (||a||=||b||=1)
// so ranking points by Euclidean distance after L2-normalizing every
// vector produces EXACTLY the same order as ranking by cosine similarity.
// CosineBallTree normalizes every point once at construction and the
// query once per call, then delegates to BallTree::query_knn unchanged --
// no new tree-pruning logic, no new correctness argument to make (BallTree
// already has one), just a change of coordinates.

struct CosineNeighborResult {
    int   index;
    float similarity; // cosine similarity, in [-1, 1]
};

std::vector<float> normalize_l2(const std::vector<float>& v);

// Cosine similarity between two arbitrary (not necessarily normalized)
// vectors -- the brute-force ground truth cosine_ann_test.cpp checks
// CosineBallTree's ranking against.
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

class CosineBallTree {
public:
    explicit CosineBallTree(const Features& points, int leaf_size = 10);

    std::vector<CosineNeighborResult> query_knn(const std::vector<float>& query, int k, bool approximate) const;

    std::size_t last_nodes_visited() const { return tree_.last_nodes_visited(); }

private:
    BallTree tree_;
};
