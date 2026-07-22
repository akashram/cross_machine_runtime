# Step 18: Compute/communication overlap — double-buffer backward pass
# with all-reduce. Portable: builds on autograd/matrix.h, data_loading's
# PrefetchQueue, and networking/ring_allreduce.

find_package(Threads REQUIRED)

add_executable(overlap_test overlap_test.cpp)
target_include_directories(overlap_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(overlap_test PRIVATE Autograd RingAllreduce Threads::Threads)

enable_testing()
add_test(NAME overlap_test COMMAND overlap_test)
