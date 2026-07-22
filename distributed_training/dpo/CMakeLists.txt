# Step 25: DPO (Direct Preference Optimization) — offline alternative to
# PPO, optimizes the policy directly on preference pairs with no reward
# model or rollouts at train time, data-parallel across simulated ranks.
# Portable: no CUDA/Linux dependency.

add_executable(dpo_test dpo_test.cpp)
target_include_directories(dpo_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dpo_test PRIVATE Transformer RingAllreduce)

enable_testing()
add_test(NAME dpo_test COMMAND dpo_test)
