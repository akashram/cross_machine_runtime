# Step 13: stacking/blending ensemble composition. Portable, builds and
# runs everywhere including this Mac.

add_library(Ensemble ensemble.cpp)
target_include_directories(Ensemble PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(Ensemble PUBLIC cxx_std_23)
set_project_warnings(Ensemble)

add_executable(ensemble_test ensemble_test.cpp)
target_link_libraries(ensemble_test PRIVATE Ensemble)
set_project_warnings(ensemble_test)

enable_testing()
add_test(NAME ensemble_test COMMAND ensemble_test)

# Real diverse-vs-correlated stacking experiment on real OpenML data --
# run manually (same convention as openml_bench/hyperparam_sweep, see
# README.md's Design note).
add_executable(ensemble_bench ensemble_bench.cpp)
target_link_libraries(ensemble_bench PRIVATE
    Ensemble
    ArffLoader
    DecisionTree
    KNN
    LinearModel
)
target_compile_definitions(ensemble_bench PRIVATE OPENML_DATA_DIR="${CMAKE_SOURCE_DIR}/ml/openml_bench/data")
target_compile_features(ensemble_bench PRIVATE cxx_std_23)
set_project_warnings(ensemble_bench)
