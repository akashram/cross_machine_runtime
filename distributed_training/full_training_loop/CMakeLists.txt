# Step 20: Full training loop — forward -> backward -> grad clip -> grad
# sync -> ZeRO optimizer step -> checkpoint, with real per-phase latency
# measurement. Composes autograd/, grad_clipping/, zero1/, and checkpoint/.

find_package(Threads REQUIRED)

add_executable(training_loop_test training_loop_test.cpp)
target_include_directories(training_loop_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(training_loop_test PRIVATE Autograd Collectives RingAllreduce Threads::Threads)

enable_testing()
add_test(NAME training_loop_test COMMAND training_loop_test)
