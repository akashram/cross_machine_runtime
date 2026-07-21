# Step 5: Distributed gradient norm clipping — portable, builds on
# networking/ring_allreduce. Builds and runs everywhere including this Mac.

add_executable(grad_clipping_test grad_clipping_test.cpp)
target_include_directories(grad_clipping_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(grad_clipping_test PRIVATE RingAllreduce)

enable_testing()
add_test(NAME grad_clipping_test COMMAND grad_clipping_test)
