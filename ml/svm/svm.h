#pragma once
#include <functional>
#include <vector>
// TODO: implement SMO (Platt 1998)

using Features = std::vector<std::vector<float>>;
using Labels   = std::vector<float>;

enum class KernelType { LINEAR, RBF, POLY };

struct SVMParams {
    KernelType kernel     = KernelType::RBF;
    float      C          = 1.0f;    // regularization
    float      gamma      = 0.0f;    // RBF kernel width (0 = 1/n_features)
    int        degree     = 3;       // poly kernel degree
    float      coef0      = 0.0f;    // poly kernel bias
    float      tol        = 1e-3f;   // SMO tolerance
    int        max_iter   = 1000;
};

class SVM {
public:
    explicit SVM(SVMParams params = {});

    // Binary classification (y ∈ {-1, +1})
    void fit(const Features& X, const Labels& y);
    Labels predict(const Features& X) const;
    float score(const Features& X, const Labels& y) const;

    int n_support_vectors() const;
};
