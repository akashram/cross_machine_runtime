# Step 18: population-based training (Jaderberg et al. 2017). Portable,
# builds and runs everywhere including this Mac.

add_library(PBT pbt.cpp)
target_include_directories(PBT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(PBT PUBLIC cxx_std_23)
set_project_warnings(PBT)

add_executable(pbt_test pbt_test.cpp)
target_link_libraries(pbt_test PRIVATE PBT)
set_project_warnings(pbt_test)

enable_testing()
add_test(NAME pbt_test COMMAND pbt_test)

# Real demo: PBT tuning LinearModel's SGD learning_rate mid-training on
# real OpenML data, backed by LinearModel::partial_fit's real warm-start
# -- run manually (same convention as the other Phase 12b/12c bench
# executables; see README.md's Design note).
add_executable(pbt_bench pbt_bench.cpp)
target_link_libraries(pbt_bench PRIVATE PBT ArffLoader LinearModel)
target_compile_definitions(pbt_bench PRIVATE OPENML_DATA_DIR="${CMAKE_SOURCE_DIR}/ml/openml_bench/data")
target_compile_features(pbt_bench PRIVATE cxx_std_23)
set_project_warnings(pbt_bench)
