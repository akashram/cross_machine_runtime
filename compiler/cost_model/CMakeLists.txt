# Step 13: Device cost model — pure C++, no MLIR dependency. Added
# unconditionally from the root CMakeLists.txt (see comment there) so it
# builds and runs on any platform, including this Mac, ahead of the rest
# of Phase 4 which needs MLIR_DIR.

add_library(CostModel CostModel.cpp)
target_include_directories(CostModel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(CostModel PUBLIC cxx_std_20)

add_executable(cost_model_bench cost_model_bench.cpp)
target_link_libraries(cost_model_bench PRIVATE CostModel)

enable_testing()
add_test(NAME cost_model_bench COMMAND cost_model_bench)
