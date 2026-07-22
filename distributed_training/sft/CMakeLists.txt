# Step 22: Supervised fine-tuning — real training over transformer/'s
# model, loss masked to response tokens, data-parallel across simulated
# ranks. Portable: no CUDA/Linux dependency.

add_executable(sft_test sft_test.cpp)
target_include_directories(sft_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sft_test PRIVATE Transformer RingAllreduce)

enable_testing()
add_test(NAME sft_test COMMAND sft_test)
