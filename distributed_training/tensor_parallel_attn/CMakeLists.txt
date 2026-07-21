# Step 12: Tensor-parallel multi-head attention. Portable: builds on
# autograd/matrix.h and networking/ring_allreduce.

add_executable(attention_test attention_test.cpp)
target_include_directories(attention_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(attention_test PRIVATE Autograd RingAllreduce)

enable_testing()
add_test(NAME attention_test COMMAND attention_test)
