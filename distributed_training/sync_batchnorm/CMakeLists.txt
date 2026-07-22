# Step 19: SyncBatchNorm — all-reduce of batch statistics across
# data-parallel ranks. Portable: builds on autograd/matrix.h and
# networking/ring_allreduce.

add_executable(sync_batchnorm_test sync_batchnorm_test.cpp)
target_include_directories(sync_batchnorm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sync_batchnorm_test PRIVATE Autograd RingAllreduce)

enable_testing()
add_test(NAME sync_batchnorm_test COMMAND sync_batchnorm_test)
