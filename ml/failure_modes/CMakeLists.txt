# Step 14: failure mode catalog -- concrete, measured demonstrations of
# each Phase 12a algorithm's known weak point. Portable, builds and runs
# everywhere including this Mac. Run manually (same convention as
# openml_bench/hyperparam_sweep/ensemble_bench -- see README.md's
# Design note); results captured into README.md.

add_executable(failure_modes_bench failure_modes_bench.cpp)
target_link_libraries(failure_modes_bench PRIVATE
    DecisionTree
    RandomForest
    GradientBoostedTrees
    SVM
    KNN
    KMeans
    PCA
    LinearModel
)
target_compile_features(failure_modes_bench PRIVATE cxx_std_23)
set_project_warnings(failure_modes_bench)
