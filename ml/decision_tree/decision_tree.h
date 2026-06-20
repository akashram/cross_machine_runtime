#pragma once
#include <vector>
#include <span>
// TODO: implement in CPU C++ (buildable on Mac)

using Features = std::vector<std::vector<float>>;  // [n_samples, n_features]
using Labels   = std::vector<float>;

struct TreeParams {
    int   max_depth        = 10;
    int   min_samples_leaf = 1;
    int   min_samples_split = 2;
    float min_impurity_decrease = 0.0f;
    enum class Criterion { GINI, ENTROPY } criterion = Criterion::GINI;
};

class DecisionTree {
public:
    explicit DecisionTree(TreeParams params = {});

    void fit(const Features& X, const Labels& y);
    Labels predict(const Features& X) const;
    float score(const Features& X, const Labels& y) const;

    int   depth() const;
    int   n_leaves() const;
    float feature_importance(int feature_idx) const;
};
