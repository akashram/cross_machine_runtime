# Step 24: PPO-based RLHF — policy (SFT init) + frozen reference + frozen
# reward model (step 23) + trained critic, clipped surrogate objective,
# KL penalty, data-parallel across simulated ranks. Portable: no
# CUDA/Linux dependency.

add_executable(ppo_rlhf_test ppo_rlhf_test.cpp)
target_include_directories(ppo_rlhf_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ppo_rlhf_test PRIVATE Transformer RingAllreduce)

enable_testing()
add_test(NAME ppo_rlhf_test COMMAND ppo_rlhf_test)
