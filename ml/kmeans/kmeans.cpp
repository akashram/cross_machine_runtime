#include "kmeans.h"

#include <algorithm>
#include <limits>
#include <random>

namespace {

float squared_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// Index of, and squared distance to, the nearest centroid.
std::pair<int, float> nearest_centroid(const std::vector<float>& point, const Features& centroids) {
    int best_idx = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (std::size_t c = 0; c < centroids.size(); ++c) {
        float d = squared_distance(point, centroids[c]);
        if (d < best_dist) {
            best_dist = d;
            best_idx = static_cast<int>(c);
        }
    }
    return {best_idx, best_dist};
}

}  // namespace

KMeans::KMeans(KMeansParams params) : params_(params) {}

// Arthur & Vassilvitskii 2007: pick the first centroid uniformly at
// random, then repeatedly pick the next centroid from the remaining
// points with probability proportional to D(x)^2, its squared distance
// to the nearest centroid chosen so far. This spreads initial centroids
// out (unlike plain random init, which can pick two centroids right next
// to each other) and gives the O(log k) approximation guarantee Lloyd's
// iteration alone doesn't have.
Features KMeans::kmeans_plus_plus_init(const Features& X) const {
    std::mt19937 rng(params_.random_state);
    std::uniform_int_distribution<std::size_t> first_pick(0, X.size() - 1);

    Features centroids;
    centroids.push_back(X[first_pick(rng)]);

    std::vector<float> min_dist_sq(X.size(), std::numeric_limits<float>::max());

    while (static_cast<int>(centroids.size()) < params_.k) {
        float total_weight = 0.0f;
        for (std::size_t i = 0; i < X.size(); ++i) {
            float d = squared_distance(X[i], centroids.back());
            min_dist_sq[i] = std::min(min_dist_sq[i], d);
            total_weight += min_dist_sq[i];
        }

        if (total_weight <= 0.0f) {
            // All remaining points coincide with an existing centroid
            // (degenerate input, e.g. fewer distinct points than k):
            // fall back to uniform choice so this always terminates.
            centroids.push_back(X[first_pick(rng)]);
            continue;
        }

        std::uniform_real_distribution<float> roll(0.0f, total_weight);
        float target = roll(rng);
        float cumulative = 0.0f;
        std::size_t chosen = X.size() - 1;
        for (std::size_t i = 0; i < X.size(); ++i) {
            cumulative += min_dist_sq[i];
            if (cumulative >= target) {
                chosen = i;
                break;
            }
        }
        centroids.push_back(X[chosen]);
    }
    return centroids;
}

// Plain uniform-random distinct-point init -- the naive baseline
// k-means++ improves on. Kept as an explicit, real alternative (not a
// stub) so tests can measure the improvement directly instead of just
// asserting it.
Features KMeans::random_init(const Features& X) const {
    std::mt19937 rng(params_.random_state);
    std::vector<int> indices(X.size());
    for (std::size_t i = 0; i < indices.size(); ++i) indices[static_cast<std::size_t>(i)] = static_cast<int>(i);
    std::shuffle(indices.begin(), indices.end(), rng);

    Features centroids;
    for (int i = 0; i < params_.k; ++i) centroids.push_back(X[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])]);
    return centroids;
}

void KMeans::fit(const Features& X) {
    int n_features = static_cast<int>(X[0].size());
    centroids_ = params_.use_kmeanspp_init ? kmeans_plus_plus_init(X) : random_init(X);
    labels_.assign(X.size(), 0);

    for (n_iter_ = 0; n_iter_ < params_.max_iter; ++n_iter_) {
        // Assignment step.
        for (std::size_t i = 0; i < X.size(); ++i) labels_[i] = nearest_centroid(X[i], centroids_).first;

        // Update step: each centroid becomes the mean of its assigned
        // points. A cluster that lost all its points keeps its old
        // centroid (a real, if rare, Lloyd's-iteration degenerate case)
        // rather than becoming NaN.
        Features new_centroids(centroids_.size(), std::vector<float>(static_cast<std::size_t>(n_features), 0.0f));
        std::vector<int> counts(centroids_.size(), 0);
        for (std::size_t i = 0; i < X.size(); ++i) {
            int c = labels_[i];
            ++counts[static_cast<std::size_t>(c)];
            for (int d = 0; d < n_features; ++d) new_centroids[static_cast<std::size_t>(c)][static_cast<std::size_t>(d)] += X[i][static_cast<std::size_t>(d)];
        }
        float max_shift = 0.0f;
        for (std::size_t c = 0; c < new_centroids.size(); ++c) {
            if (counts[c] == 0) {
                new_centroids[c] = centroids_[c];
                continue;
            }
            for (int d = 0; d < n_features; ++d) new_centroids[c][static_cast<std::size_t>(d)] /= static_cast<float>(counts[c]);
            max_shift = std::max(max_shift, squared_distance(new_centroids[c], centroids_[c]));
        }
        centroids_ = std::move(new_centroids);

        if (max_shift < params_.tol * params_.tol) {
            ++n_iter_;
            break;
        }
    }

    inertia_ = 0.0f;
    for (std::size_t i = 0; i < X.size(); ++i) inertia_ += squared_distance(X[i], centroids_[static_cast<std::size_t>(labels_[i])]);
}

std::vector<int> KMeans::predict(const Features& X) const {
    std::vector<int> out;
    out.reserve(X.size());
    for (const auto& row : X) out.push_back(nearest_centroid(row, centroids_).first);
    return out;
}

std::vector<float> elbow_curve(const Features& X, int max_k, unsigned random_state) {
    std::vector<float> inertias;
    for (int k = 1; k <= max_k; ++k) {
        KMeansParams p;
        p.k = k;
        p.random_state = random_state;
        KMeans km(p);
        km.fit(X);
        inertias.push_back(km.inertia());
    }
    return inertias;
}
