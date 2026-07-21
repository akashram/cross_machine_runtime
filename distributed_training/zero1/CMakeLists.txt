# Step 7: ZeRO Stage 1 — shard Adam optimizer state across data-parallel
# ranks. Portable: builds on autograd/ (toy model + flatten/unflatten),
# networking/ring_allreduce (gradient all-reduce), and
# networking/collectives (parameter all-gather).

add_executable(zero1_test zero1_test.cpp)
target_include_directories(zero1_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(zero1_test PRIVATE Autograd Collectives RingAllreduce)

enable_testing()
add_test(NAME zero1_test COMMAND zero1_test)
