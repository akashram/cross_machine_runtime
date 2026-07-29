# Step 15: Bayesian optimization -- GP surrogate (Cholesky-based exact
# inference), Expected Improvement / UCB acquisition. Portable, builds
# and runs everywhere including this Mac.

add_library(BayesianOpt gp.cpp bayesian_opt.cpp)
target_include_directories(BayesianOpt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(BayesianOpt PUBLIC cxx_std_23)
set_project_warnings(BayesianOpt)

add_executable(bayesian_opt_test bayesian_opt_test.cpp)
target_link_libraries(bayesian_opt_test PRIVATE BayesianOpt)
set_project_warnings(bayesian_opt_test)

enable_testing()
add_test(NAME bayesian_opt_test COMMAND bayesian_opt_test)

# Real hyperparameter-tuning benchmark (BO vs random search vs grid
# search) on real OpenML data -- run manually (same convention as
# openml_bench/hyperparam_sweep/ensemble_bench/failure_modes_bench; see
# README.md's Design note).
add_executable(bayesian_opt_bench bayesian_opt_bench.cpp)
target_link_libraries(bayesian_opt_bench PRIVATE BayesianOpt ArffLoader SVM)
target_compile_definitions(bayesian_opt_bench PRIVATE OPENML_DATA_DIR="${CMAKE_SOURCE_DIR}/ml/openml_bench/data")
target_compile_features(bayesian_opt_bench PRIVATE cxx_std_23)
set_project_warnings(bayesian_opt_bench)
