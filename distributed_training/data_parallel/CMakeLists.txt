# Step 3: Data parallel training baseline — portable, built on
# networking/ring_allreduce (which is itself built on networking/common's
# Channel). Builds and runs everywhere including this Mac.

add_executable(data_parallel_test data_parallel_test.cpp)
target_include_directories(data_parallel_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(data_parallel_test PRIVATE RingAllreduce)

enable_testing()
add_test(NAME data_parallel_test COMMAND data_parallel_test)
