#pragma once
#include <cstddef>
#include <memory>
#include <vector>

// PLAN.md Phase 12a step 5: k-NN via KD-tree (exact) and ball tree
// (approximate), SIMD distance computation.
//
// "SIMD distance computation" follows the same convention as
// decision_tree/svm: squared_distance() below is a single linear pass
// over contiguous float arrays, auto-vectorizable at -O3 (no
// hand-written AVX intrinsics -- see README.md's Design note, same
// honest caveat as decision_tree's).

using Features = std::vector<std::vector<float>>;
using Labels   = std::vector<float>;

// Squared Euclidean distance. Contiguous linear accumulation loop --
// the auto-vectorizable shape, not a hand-tuned SIMD kernel.
float squared_distance(const std::vector<float>& a, const std::vector<float>& b);

struct NeighborResult {
    int   index;
    float dist_sq;
};

// Exact k-NN via a classic depth-round-robin-split KD-tree with
// branch-and-bound pruning (Bentley 1975): the far child is only
// visited if the splitting-hyperplane distance could still beat the
// current k-th best, so this always returns the true k nearest
// neighbors.
class KDTree {
public:
    explicit KDTree(const Features& points);

    std::vector<NeighborResult> query_knn(const std::vector<float>& query, int k) const;

    // Nodes visited by the most recent query_knn() call -- lets tests/
    // benchmarks measure how much branch-and-bound pruning actually cut
    // vs. a brute-force O(n) scan.
    std::size_t last_nodes_visited() const { return last_nodes_visited_; }

private:
    struct Node {
        int split_dim = 0;
        int point_idx = -1;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    Features points_;
    std::unique_ptr<Node> root_;
    mutable std::size_t last_nodes_visited_ = 0;

    std::unique_ptr<Node> build(std::vector<int>& indices, int lo, int hi, int depth);
    void search(const Node* node, const std::vector<float>& query, int k,
                std::vector<NeighborResult>& heap) const;
};

// Ball tree (Omohundro 1989): built once, queryable in two modes on the
// identical tree structure so the two can be compared directly:
//   - exact: bound-and-backtrack, same correctness guarantee as KDTree,
//     using the triangle-inequality radius bound instead of an
//     axis-aligned hyperplane -- what makes ball trees not degrade to
//     brute force in high dimensions the way KD-trees tend to.
//   - approximate (defeatist search, Liu et al. 2004 IT/ball-tree ANN
//     literature): descend straight to one leaf via nearest-child
//     choice with NO backtracking at all, then brute-force the k
//     nearest within that single leaf. O(log n) node visits, but not
//     guaranteed to find the true k nearest -- this is PLAN.md's "ball
//     tree (approximate)".
class BallTree {
public:
    explicit BallTree(const Features& points, int leaf_size = 10);

    std::vector<NeighborResult> query_knn(const std::vector<float>& query, int k, bool approximate) const;

    std::size_t last_nodes_visited() const { return last_nodes_visited_; }

private:
    struct Node {
        std::vector<float> center;
        float radius = 0.0f;
        std::vector<int> point_indices;  // non-empty only at leaves
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        bool is_leaf() const { return left == nullptr && right == nullptr; }
    };

    Features points_;
    int leaf_size_;
    std::unique_ptr<Node> root_;
    mutable std::size_t last_nodes_visited_ = 0;

    std::unique_ptr<Node> build(std::vector<int>& indices, int lo, int hi);
    void search_exact(const Node* node, const std::vector<float>& query, int k,
                       std::vector<NeighborResult>& heap) const;
    void search_defeatist(const Node* node, const std::vector<float>& query, int k,
                           std::vector<NeighborResult>& heap) const;
};

enum class NeighborStructure { KD_TREE, BALL_TREE };

struct KNNParams {
    int k = 5;
    NeighborStructure structure = NeighborStructure::KD_TREE;
    bool approximate = false;  // only meaningful when structure == BALL_TREE
};

class KNNClassifier {
public:
    explicit KNNClassifier(KNNParams params = {});

    void fit(const Features& X, const Labels& y);
    Labels predict(const Features& X) const;
    float score(const Features& X, const Labels& y) const;

private:
    KNNParams params_;
    Features X_train_;
    Labels y_train_;
    std::unique_ptr<KDTree> kd_tree_;
    std::unique_ptr<BallTree> ball_tree_;

    std::vector<NeighborResult> neighbors(const std::vector<float>& query) const;
};
