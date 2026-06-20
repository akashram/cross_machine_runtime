#pragma once
#include "decision_tree/decision_tree.h"
#include <cstddef>
// TODO: implement with work-stealing parallelism from foundation/

struct RFParams {
    int    n_estimators    = 100;
    int    max_depth       = -1;  // unlimited
    int    min_samples_leaf = 1;
    float  max_features    = -1.0f;  // sqrt(n_features) if -1
    int    random_state    = 42;
};

class RandomForest {
public:
    explicit RandomForest(RFParams params = {});

    void fit(const Features& X, const Labels& y);
    Labels predict(const Features& X) const;
    float score(const Features& X, const Labels& y) const;

    float oob_error() const;
    std::vector<float> feature_importances() const;

private:
    // TODO: vector of DecisionTree, parallel fit via work-stealing pool
};
