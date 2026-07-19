# Step 13: Tree all-reduce — portable, builds and runs everywhere
# including this Mac. tree_broadcast is reused by collectives/ (step 14).

add_library(TreeAllreduce tree_allreduce.cpp)
target_link_libraries(TreeAllreduce PUBLIC NetCommon)
target_include_directories(TreeAllreduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(tree_allreduce_test tree_allreduce_test.cpp)
target_link_libraries(tree_allreduce_test PRIVATE TreeAllreduce)

enable_testing()
add_test(NAME tree_allreduce_test COMMAND tree_allreduce_test)
