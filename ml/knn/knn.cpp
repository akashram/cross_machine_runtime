#include "knn.h"

#include <algorithm>
#include <cmath>
#include <map>

float squared_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

namespace {

bool by_dist_less(const NeighborResult& a, const NeighborResult& b) { return a.dist_sq < b.dist_sq; }

// Bounded max-heap of the k closest candidates seen so far: push while
// under capacity, otherwise replace the current worst (heap top) only
// if the new candidate is strictly closer. Shared by KDTree and both
// BallTree search modes.
void offer(std::vector<NeighborResult>& heap, int k, int idx, float dist_sq) {
    if (static_cast<int>(heap.size()) < k) {
        heap.push_back({idx, dist_sq});
        std::push_heap(heap.begin(), heap.end(), by_dist_less);
    } else if (dist_sq < heap.front().dist_sq) {
        std::pop_heap(heap.begin(), heap.end(), by_dist_less);
        heap.back() = {idx, dist_sq};
        std::push_heap(heap.begin(), heap.end(), by_dist_less);
    }
}

std::vector<NeighborResult> heap_to_sorted(std::vector<NeighborResult> heap) {
    std::sort(heap.begin(), heap.end(), by_dist_less);
    return heap;
}

}  // namespace

// ---------------------------------------------------------------------
// KDTree
// ---------------------------------------------------------------------

KDTree::KDTree(const Features& points) : points_(points) {
    std::vector<int> indices(points_.size());
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int>(i);
    root_ = build(indices, 0, static_cast<int>(indices.size()), 0);
}

std::unique_ptr<KDTree::Node> KDTree::build(std::vector<int>& indices, int lo, int hi, int depth) {
    if (lo >= hi) return nullptr;
    int n_features = static_cast<int>(points_[0].size());
    int split_dim = depth % n_features;
    int mid = lo + (hi - lo) / 2;

    std::nth_element(indices.begin() + lo, indices.begin() + mid, indices.begin() + hi,
                      [&](int a, int b) { return points_[static_cast<std::size_t>(a)][static_cast<std::size_t>(split_dim)] <
                                                 points_[static_cast<std::size_t>(b)][static_cast<std::size_t>(split_dim)]; });

    auto node = std::make_unique<Node>();
    node->split_dim = split_dim;
    node->point_idx = indices[static_cast<std::size_t>(mid)];
    node->left = build(indices, lo, mid, depth + 1);
    node->right = build(indices, mid + 1, hi, depth + 1);
    return node;
}

void KDTree::search(const Node* node, const std::vector<float>& query, int k,
                     std::vector<NeighborResult>& heap) const {
    if (!node) return;
    ++last_nodes_visited_;

    float d = squared_distance(points_[static_cast<std::size_t>(node->point_idx)], query);
    offer(heap, k, node->point_idx, d);

    float diff = query[static_cast<std::size_t>(node->split_dim)] -
                 points_[static_cast<std::size_t>(node->point_idx)][static_cast<std::size_t>(node->split_dim)];
    const Node* near = diff < 0.0f ? node->left.get() : node->right.get();
    const Node* far = diff < 0.0f ? node->right.get() : node->left.get();

    search(near, query, k, heap);

    // The far subtree can only contain a closer point if the query's
    // distance to the splitting hyperplane is itself smaller than the
    // current k-th best -- the branch-and-bound prune that keeps this
    // from degrading to a brute-force O(n) scan.
    float plane_dist_sq = diff * diff;
    if (static_cast<int>(heap.size()) < k || plane_dist_sq < heap.front().dist_sq) {
        search(far, query, k, heap);
    }
}

std::vector<NeighborResult> KDTree::query_knn(const std::vector<float>& query, int k) const {
    last_nodes_visited_ = 0;
    std::vector<NeighborResult> heap;
    heap.reserve(static_cast<std::size_t>(k));
    search(root_.get(), query, k, heap);
    return heap_to_sorted(std::move(heap));
}

// ---------------------------------------------------------------------
// BallTree
// ---------------------------------------------------------------------

BallTree::BallTree(const Features& points, int leaf_size) : points_(points), leaf_size_(leaf_size) {
    std::vector<int> indices(points_.size());
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int>(i);
    root_ = build(indices, 0, static_cast<int>(indices.size()));
}

std::unique_ptr<BallTree::Node> BallTree::build(std::vector<int>& indices, int lo, int hi) {
    auto node = std::make_unique<Node>();
    int n_features = static_cast<int>(points_[0].size());
    int count = hi - lo;

    // Centroid and radius over the full [lo, hi) range: computed
    // directly (not derived from children) so the bound is valid for
    // pruning regardless of how the split below partitions points.
    node->center.assign(static_cast<std::size_t>(n_features), 0.0f);
    for (int i = lo; i < hi; ++i)
        for (int d = 0; d < n_features; ++d)
            node->center[static_cast<std::size_t>(d)] += points_[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])][static_cast<std::size_t>(d)];
    for (int d = 0; d < n_features; ++d) node->center[static_cast<std::size_t>(d)] /= static_cast<float>(count);

    node->radius = 0.0f;
    for (int i = lo; i < hi; ++i) {
        float d = std::sqrt(squared_distance(node->center, points_[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])]));
        node->radius = std::max(node->radius, d);
    }

    if (count <= leaf_size_) {
        node->point_indices.assign(indices.begin() + lo, indices.begin() + hi);
        return node;
    }

    // Two-pivot split (Omohundro 1989 / Uhlmann 1991): find the point
    // farthest from the centroid (p1), then the point farthest from p1
    // (p2) -- a cheap proxy for the widest axis of the point cloud in
    // this subtree -- and partition by which pivot each point is
    // nearer to.
    int p1 = indices[static_cast<std::size_t>(lo)];
    float best = -1.0f;
    for (int i = lo; i < hi; ++i) {
        float d = squared_distance(node->center, points_[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])]);
        if (d > best) { best = d; p1 = indices[static_cast<std::size_t>(i)]; }
    }
    int p2 = indices[static_cast<std::size_t>(lo)];
    best = -1.0f;
    for (int i = lo; i < hi; ++i) {
        float d = squared_distance(points_[static_cast<std::size_t>(p1)], points_[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])]);
        if (d > best) { best = d; p2 = indices[static_cast<std::size_t>(i)]; }
    }

    auto mid_it = std::partition(indices.begin() + lo, indices.begin() + hi, [&](int idx) {
        float d1 = squared_distance(points_[static_cast<std::size_t>(p1)], points_[static_cast<std::size_t>(idx)]);
        float d2 = squared_distance(points_[static_cast<std::size_t>(p2)], points_[static_cast<std::size_t>(idx)]);
        return d1 <= d2;
    });
    int mid = static_cast<int>(mid_it - indices.begin());

    // Degenerate case (e.g. every point equidistant, or all nearer to
    // one pivot): fall back to a plain index-range median split so
    // recursion always makes progress on both sides.
    if (mid == lo || mid == hi) mid = lo + (hi - lo) / 2;

    node->left = build(indices, lo, mid);
    node->right = build(indices, mid, hi);
    return node;
}

void BallTree::search_exact(const Node* node, const std::vector<float>& query, int k,
                             std::vector<NeighborResult>& heap) const {
    if (!node) return;

    // Triangle-inequality lower bound: no point in this ball can be
    // closer to the query than dist(query, center) - radius. If that
    // already loses to the current k-th best, prune the whole subtree.
    float dist_to_center = std::sqrt(squared_distance(node->center, query));
    float lower_bound = dist_to_center - node->radius;
    float lower_bound_sq = lower_bound > 0.0f ? lower_bound * lower_bound : 0.0f;
    if (static_cast<int>(heap.size()) >= k && lower_bound_sq > heap.front().dist_sq) return;

    ++last_nodes_visited_;

    if (node->is_leaf()) {
        for (int idx : node->point_indices) offer(heap, k, idx, squared_distance(points_[static_cast<std::size_t>(idx)], query));
        return;
    }

    float dist_left = squared_distance(node->left->center, query);
    float dist_right = squared_distance(node->right->center, query);
    const Node* near = dist_left <= dist_right ? node->left.get() : node->right.get();
    const Node* far = dist_left <= dist_right ? node->right.get() : node->left.get();

    search_exact(near, query, k, heap);
    search_exact(far, query, k, heap);
}

void BallTree::search_defeatist(const Node* node, const std::vector<float>& query, int k,
                                 std::vector<NeighborResult>& heap) const {
    ++last_nodes_visited_;
    if (node->is_leaf()) {
        for (int idx : node->point_indices) offer(heap, k, idx, squared_distance(points_[static_cast<std::size_t>(idx)], query));
        return;
    }

    // Defeatist descent: go straight to the nearer child, never look at
    // the other subtree at all -- the whole reason this is an
    // approximate rather than exact algorithm.
    float dist_left = squared_distance(node->left->center, query);
    float dist_right = squared_distance(node->right->center, query);
    search_defeatist(dist_left <= dist_right ? node->left.get() : node->right.get(), query, k, heap);
}

std::vector<NeighborResult> BallTree::query_knn(const std::vector<float>& query, int k, bool approximate) const {
    last_nodes_visited_ = 0;
    std::vector<NeighborResult> heap;
    heap.reserve(static_cast<std::size_t>(k));
    if (approximate)
        search_defeatist(root_.get(), query, k, heap);
    else
        search_exact(root_.get(), query, k, heap);
    return heap_to_sorted(std::move(heap));
}

// ---------------------------------------------------------------------
// KNNClassifier
// ---------------------------------------------------------------------

KNNClassifier::KNNClassifier(KNNParams params) : params_(params) {}

void KNNClassifier::fit(const Features& X, const Labels& y) {
    X_train_ = X;
    y_train_ = y;
    kd_tree_.reset();
    ball_tree_.reset();
    if (params_.structure == NeighborStructure::KD_TREE)
        kd_tree_ = std::make_unique<KDTree>(X_train_);
    else
        ball_tree_ = std::make_unique<BallTree>(X_train_);
}

std::vector<NeighborResult> KNNClassifier::neighbors(const std::vector<float>& query) const {
    if (params_.structure == NeighborStructure::KD_TREE)
        return kd_tree_->query_knn(query, params_.k);
    return ball_tree_->query_knn(query, params_.k, params_.approximate);
}

Labels KNNClassifier::predict(const Features& X) const {
    Labels out;
    out.reserve(X.size());
    for (const auto& row : X) {
        std::vector<NeighborResult> nn = neighbors(row);

        // Majority vote; ties broken in favor of whichever tied class
        // contains the single nearest neighbor (nn is distance-sorted).
        std::map<float, int> votes;
        for (const auto& n : nn) ++votes[y_train_[static_cast<std::size_t>(n.index)]];

        float best_label = nn.empty() ? 0.0f : y_train_[static_cast<std::size_t>(nn.front().index)];
        int best_count = -1;
        for (const auto& n : nn) {
            float label = y_train_[static_cast<std::size_t>(n.index)];
            int count = votes[label];
            if (count > best_count) {
                best_count = count;
                best_label = label;
            }
        }
        out.push_back(best_label);
    }
    return out;
}

float KNNClassifier::score(const Features& X, const Labels& y) const {
    if (y.empty()) return 0.0f;
    Labels pred = predict(X);
    int correct = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        if (pred[i] == y[i]) ++correct;
    return static_cast<float>(correct) / static_cast<float>(y.size());
}
