# Step 9: ZeRO Stage 3 — shard parameters in addition to gradients and
# optimizer state. Portable: builds on autograd/, zero1/adam.h, and
# networking/collectives.

add_executable(zero3_test zero3_test.cpp)
target_include_directories(zero3_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(zero3_test PRIVATE Autograd Collectives RingAllreduce)

enable_testing()
add_test(NAME zero3_test COMMAND zero3_test)
