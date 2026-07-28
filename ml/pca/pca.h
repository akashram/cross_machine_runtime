#pragma once
#include <vector>

// PLAN.md Phase 12a step 7: PCA via randomized SVD (Halko, Martinsson &
// Tropp 2011), explained variance ratio, whitening.

using Features = std::vector<std::vector<float>>;  // [n_samples, n_features]

struct PCAParams {
    int n_components      = 2;
    int n_oversamples      = 10;  // Halko et al.'s oversampling parameter (l = n_components + n_oversamples)
    int n_power_iterations = 2;   // subspace power iterations -- sharpens the spectrum when singular values decay slowly
    bool whiten            = false;
    unsigned random_state  = 0;
};

class PCA {
public:
    explicit PCA(PCAParams params = {});

    void fit(const Features& X);
    Features transform(const Features& X) const;
    Features fit_transform(const Features& X);

    const Features& components() const { return components_; }  // rows are principal axes, [n_components, n_features]
    const std::vector<float>& singular_values() const { return singular_values_; }
    const std::vector<float>& explained_variance_ratio() const { return explained_variance_ratio_; }

private:
    PCAParams params_;
    std::vector<float> mean_;
    Features components_;
    std::vector<float> singular_values_;
    std::vector<float> explained_variance_;
    std::vector<float> explained_variance_ratio_;
};
