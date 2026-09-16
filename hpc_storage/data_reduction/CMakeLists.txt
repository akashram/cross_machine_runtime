# Phase 21 step 7: data reduction effectiveness on real AI-workload
# artifacts. Trains the real transformer/ model; measures with real zlib.

find_package(ZLIB REQUIRED)

add_executable(data_reduction_test data_reduction_test.cpp)
target_link_libraries(data_reduction_test PRIVATE Transformer ZLIB::ZLIB)
set_project_warnings(data_reduction_test)

enable_testing()
add_test(NAME data_reduction_test COMMAND data_reduction_test)
