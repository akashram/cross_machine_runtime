# Step 16: MoE / expert parallelism — top-1 routing + variable-size
# all-to-all dispatch/combine. Portable: builds on autograd/matrix.h and
# networking/common (raw Channel send/recv, no CUDA/Linux dependency).

add_executable(moe_test moe_test.cpp)
target_include_directories(moe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(moe_test PRIVATE Autograd NetCommon)

enable_testing()
add_test(NAME moe_test COMMAND moe_test)
