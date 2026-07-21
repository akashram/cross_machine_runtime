# Step 13: Sequence parallelism, interleaved with tensor parallelism.
# Portable: builds on autograd/matrix.h, col_row_linear's tensor-parallel
# linear layers, and networking/collectives.

add_executable(seq_parallel_test seq_parallel_test.cpp)
target_include_directories(seq_parallel_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(seq_parallel_test PRIVATE Autograd Collectives RingAllreduce)

enable_testing()
add_test(NAME seq_parallel_test COMMAND seq_parallel_test)
