# Step 17: Hyperband / ASHA -- successive halving, early stopping,
# asynchronous promotion. Portable, builds and runs everywhere including
# this Mac.

add_library(Hyperband hyperband.cpp)
target_include_directories(Hyperband PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(Hyperband PUBLIC cxx_std_23)
set_project_warnings(Hyperband)

add_executable(hyperband_test hyperband_test.cpp)
target_link_libraries(hyperband_test PRIVATE Hyperband)
set_project_warnings(hyperband_test)

enable_testing()
add_test(NAME hyperband_test COMMAND hyperband_test)

# Real GBT hyperparameter tuning benchmark (resource = n_estimators),
# measuring total training cost saved by early stopping -- run manually
# (same convention as the other Phase 12b/12c bench executables; see
# README.md's Design note).
add_executable(hyperband_bench hyperband_bench.cpp)
target_link_libraries(hyperband_bench PRIVATE Hyperband ArffLoader GradientBoostedTrees)
target_compile_definitions(hyperband_bench PRIVATE OPENML_DATA_DIR="${CMAKE_SOURCE_DIR}/ml/openml_bench/data")
target_compile_features(hyperband_bench PRIVATE cxx_std_23)
set_project_warnings(hyperband_bench)
