# Step 23: reward model training — Bradley-Terry preference loss over
# transformer/'s model (reused body + scalar reward head), data-parallel
# across simulated ranks. Portable: no CUDA/Linux dependency.

add_executable(reward_model_test reward_model_test.cpp)
target_include_directories(reward_model_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(reward_model_test PRIVATE Transformer RingAllreduce)

enable_testing()
add_test(NAME reward_model_test COMMAND reward_model_test)
