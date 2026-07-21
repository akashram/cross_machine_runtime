# Step 15: 3D parallelism (data + tensor + pipeline composition). Portable:
# builds on autograd/matrix.h, col_row_linear's tensor-parallel linear
# layers, and networking/ring_allreduce.

add_executable(parallel_3d_test parallel_3d_test.cpp)
target_include_directories(parallel_3d_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(parallel_3d_test PRIVATE Autograd RingAllreduce)

enable_testing()
add_test(NAME parallel_3d_test COMMAND parallel_3d_test)
