# Step 9: OpenML CC-18 runner. Portable, builds and runs everywhere
# including this Mac -- no Python/sklearn dependency (see README's
# Design note: datasets were fetched once from the real OpenML REST API
# and committed as ARFF files under data/, parsed here by a small
# hand-written loader).

add_library(ArffLoader arff_loader.cpp)
target_include_directories(ArffLoader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ArffLoader PUBLIC cxx_std_23)
set_project_warnings(ArffLoader)

add_executable(arff_loader_test arff_loader_test.cpp)
target_link_libraries(arff_loader_test PRIVATE ArffLoader)
target_compile_definitions(arff_loader_test PRIVATE OPENML_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
set_project_warnings(arff_loader_test)

enable_testing()
add_test(NAME arff_loader_test COMMAND arff_loader_test)

# The full 6-algorithm x 8-dataset sweep -- a benchmark to be run
# manually (same convention as cpu_engine/bench: add_executable, not
# add_test, since a full sweep is slower than a routine `ctest` pass
# should take). Results captured into README.md.
add_executable(openml_bench openml_bench.cpp)
target_link_libraries(openml_bench PRIVATE
    ArffLoader
    DecisionTree
    RandomForest
    GradientBoostedTrees
    SVM
    KNN
    LinearModel
)
target_compile_definitions(openml_bench PRIVATE OPENML_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_compile_features(openml_bench PRIVATE cxx_std_23)
set_project_warnings(openml_bench)
