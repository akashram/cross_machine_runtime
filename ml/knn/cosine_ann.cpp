#include "cosine_ann.h"

#include <algorithm>
#include <cmath>

std::vector<float> normalize_l2(const std::vector<float>& v) {
    float norm_sq = 0.0f;
    for (float x : v) norm_sq += x * x;
    float norm = std::sqrt(std::max(norm_sq, 1e-12f));
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = v[i] / norm;
    return out;
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a_sq = 0.0f, norm_b_sq = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    float denom = std::sqrt(std::max(norm_a_sq, 1e-12f)) * std::sqrt(std::max(norm_b_sq, 1e-12f));
    return dot / denom;
}

namespace {
Features normalize_all(const Features& points) {
    Features out;
    out.reserve(points.size());
    for (const auto& p : points) out.push_back(normalize_l2(p));
    return out;
}
}  // namespace

CosineBallTree::CosineBallTree(const Features& points, int leaf_size)
    : tree_(normalize_all(points), leaf_size) {}

std::vector<CosineNeighborResult> CosineBallTree::query_knn(const std::vector<float>& query, int k,
                                                             bool approximate) const {
    std::vector<float> normalized_query = normalize_l2(query);
    std::vector<NeighborResult> raw = tree_.query_knn(normalized_query, k, approximate);

    std::vector<CosineNeighborResult> out;
    out.reserve(raw.size());
    for (const auto& r : raw) {
        // cos(a,b) = 1 - ||a-b||^2 / 2 for unit a, b -- see knn's own
        // derivation in cosine_ann.h.
        float similarity = 1.0f - r.dist_sq * 0.5f;
        out.push_back({r.index, similarity});
    }
    return out;
}
