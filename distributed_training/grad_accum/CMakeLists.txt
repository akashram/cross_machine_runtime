# Step 4: Gradient accumulation — portable, builds on data_parallel's
# linreg.h and networking/ring_allreduce. Builds and runs everywhere
# including this Mac.

add_executable(grad_accum_test grad_accum_test.cpp)
target_include_directories(grad_accum_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(grad_accum_test PRIVATE RingAllreduce)

enable_testing()
add_test(NAME grad_accum_test COMMAND grad_accum_test)
