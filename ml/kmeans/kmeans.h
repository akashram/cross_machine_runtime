#pragma once
#include <vector>

// PLAN.md Phase 12a step 6: k-means++ (Arthur & Vassilvitskii 2007)
// initialization + Lloyd's iteration, elbow method for k selection.

using Features = std::vector<std::vector<float>>;

struct KMeansParams {
    int k              = 3;
    int max_iter       = 300;
    float tol          = 1e-4f;  // converged once max centroid movement (L2) drops below this
    unsigned random_state = 0;

    // true (default): Arthur & Vassilvitskii 2007 k-means++ init. false:
    // plain uniform-random distinct-point init, kept only so tests can
    // measure k-means++'s actual benefit against a real baseline rather
    // than asserting it.
    bool use_kmeanspp_init = true;
};

class KMeans {
public:
    explicit KMeans(KMeansParams params = {});

    void fit(const Features& X);
    std::vector<int> predict(const Features& X) const;

    const std::vector<int>& labels() const { return labels_; }        // training-set assignments from fit()
    const Features& centroids() const { return centroids_; }
    float inertia() const { return inertia_; }                        // sum of squared distances to assigned centroid
    int n_iter() const { return n_iter_; }

private:
    KMeansParams params_;
    Features centroids_;
    std::vector<int> labels_;
    float inertia_ = 0.0f;
    int n_iter_ = 0;

    Features kmeans_plus_plus_init(const Features& X) const;
    Features random_init(const Features& X) const;
};

// Elbow method: fit KMeans for k = 1..max_k on the same data, return the
// resulting inertia at each k (index 0 == k=1). Callers look for the
// "elbow" -- the k past which inertia stops dropping sharply.
std::vector<float> elbow_curve(const Features& X, int max_k, unsigned random_state = 0);
