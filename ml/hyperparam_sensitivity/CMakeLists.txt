# Step 12: hyperparameter sensitivity sweeps across every Phase 12a
# algorithm, run manually against real OpenML data reused from
# ml/openml_bench (same convention as that step: add_executable, not
# add_test -- see README.md's Design note).

add_executable(hyperparam_sweep hyperparam_sweep.cpp)
target_link_libraries(hyperparam_sweep PRIVATE
    ArffLoader
    DecisionTree
    RandomForest
    GradientBoostedTrees
    SVM
    KNN
    KMeans
    PCA
    LinearModel
)
target_include_directories(hyperparam_sweep PRIVATE ${CMAKE_SOURCE_DIR}/ml/openml_bench)
target_compile_definitions(hyperparam_sweep PRIVATE OPENML_DATA_DIR="${CMAKE_SOURCE_DIR}/ml/openml_bench/data")
target_compile_features(hyperparam_sweep PRIVATE cxx_std_23)
set_project_warnings(hyperparam_sweep)
