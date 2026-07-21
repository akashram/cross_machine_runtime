# Step 11: Column/row-parallel tensor-parallel linear layers. Portable:
# builds on autograd/matrix.h and networking/ring_allreduce.

add_executable(tensor_parallel_linear_test tensor_parallel_linear_test.cpp)
target_include_directories(tensor_parallel_linear_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tensor_parallel_linear_test PRIVATE Autograd RingAllreduce)

enable_testing()
add_test(NAME tensor_parallel_linear_test COMMAND tensor_parallel_linear_test)
