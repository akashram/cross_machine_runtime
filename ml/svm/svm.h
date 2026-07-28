#pragma once
#include <functional>
#include <vector>

// PLAN.md Phase 12a step 4: Platt (1998) SMO -- the classic "simplified
// SMO" variant (Platt's paper section 12.2 / the commonly taught CS229
// pseudocode): random second-multiplier selection rather than Platt's
// full heuristic second-choice/cache bookkeeping, which trades a modest
// amount of convergence speed for a much simpler, still-correct
// implementation. A real, scoped-out follow-up (not implicitly claimed
// here), same spirit as decision_tree's "no hand AVX intrinsics" caveat.

using Features = std::vector<std::vector<float>>;
using Labels   = std::vector<float>;

enum class KernelType { LINEAR, RBF, POLY };

struct SVMParams {
    KernelType kernel     = KernelType::RBF;
    float      C          = 1.0f;    // regularization
    float      gamma      = 0.0f;    // RBF/poly kernel width (0 = 1/n_features)
    int        degree     = 3;       // poly kernel degree
    float      coef0      = 0.0f;    // poly kernel bias
    float      tol        = 1e-3f;   // SMO KKT-violation tolerance
    int        max_iter   = 1000;    // max outer SMO passes
};

class SVM {
public:
    explicit SVM(SVMParams params = {});

    // Binary classification (y in {-1, +1})
    void fit(const Features& X, const Labels& y);
    Labels predict(const Features& X) const;
    float score(const Features& X, const Labels& y) const;

    int n_support_vectors() const { return static_cast<int>(X_sv_.size()); }

private:
    SVMParams params_;
    Features X_sv_;              // support vectors only (alpha_i > 0), kept after fit()
    Labels y_sv_;
    std::vector<float> alpha_sv_;
    float b_ = 0.0f;
    float effective_gamma_ = 1.0f;  // params_.gamma resolved against n_features at fit() time

    float kernel(const std::vector<float>& a, const std::vector<float>& b) const;
};
