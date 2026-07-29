# Step 16: TPE (Tree-structured Parzen Estimator, Bergstra et al. 2011).
# Portable, builds and runs everywhere including this Mac.

add_library(TPE tpe.cpp)
target_include_directories(TPE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(TPE PUBLIC cxx_std_23)
set_project_warnings(TPE)

add_executable(tpe_test tpe_test.cpp)
target_link_libraries(tpe_test PRIVATE TPE)
set_project_warnings(tpe_test)

enable_testing()
add_test(NAME tpe_test COMMAND tpe_test)

# Real hyperparameter-tuning benchmark (TPE vs GP-based BO from
# ml/bayesian_opt vs random search) on real OpenML data -- run manually
# (same convention as the other Phase 12b/12c bench executables; see
# README.md's Design note).
add_executable(tpe_bench tpe_bench.cpp)
target_link_libraries(tpe_bench PRIVATE TPE BayesianOpt ArffLoader SVM)
target_compile_definitions(tpe_bench PRIVATE OPENML_DATA_DIR="${CMAKE_SOURCE_DIR}/ml/openml_bench/data")
target_compile_features(tpe_bench PRIVATE cxx_std_23)
set_project_warnings(tpe_bench)
