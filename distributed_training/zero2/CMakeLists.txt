# Step 8: ZeRO Stage 2 — shard gradients in addition to optimizer state.
# Portable: builds on autograd/, zero1/adam.h, and networking/collectives.

add_executable(zero2_test zero2_test.cpp)
target_include_directories(zero2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(zero2_test PRIVATE Autograd Collectives RingAllreduce)

enable_testing()
add_test(NAME zero2_test COMMAND zero2_test)
