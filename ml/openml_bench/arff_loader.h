#pragma once
#include <string>
#include <vector>

// PLAN.md Phase 12a step 9: OpenML CC-18 runner. Loads real datasets
// fetched from OpenML (https://www.openml.org/api/v1/json/study/99 is
// the CC-18 suite listing) and committed as ARFF files under data/ --
// see README.md's Design note for which 8 of the suite's 72 datasets
// were selected and why, and for this loader's honest scope limits
// (numeric feature attributes + one final nominal class attribute only;
// no categorical-feature one-hot encoding, no sparse ARFF).

using Features = std::vector<std::vector<float>>;
using Labels   = std::vector<float>;

struct OpenMLDataset {
    std::string name;
    Features X;
    Labels y;
    std::vector<std::string> class_names;  // declaration order; y values are indices into this
};

OpenMLDataset load_arff(const std::string& path, const std::string& display_name);

struct TrainTestSplit {
    Features X_train, X_test;
    Labels y_train, y_test;
};

// Shuffled train/test split with a fixed seed for reproducibility.
TrainTestSplit split_train_test(const Features& X, const Labels& y, float test_fraction, unsigned random_state);
