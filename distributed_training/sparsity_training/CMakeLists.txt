# Step 21: 2:4 structured sparsity during training. Portable: builds on
# autograd/. Real 2:4 Tensor Core throughput is hardware-gated in
# gpu_engine/sparsity/ (Phase 3) — this step validates training-time
# convergence with the mask applied, not raw throughput.

add_executable(sparsity_training_test sparsity_training_test.cpp)
target_include_directories(sparsity_training_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sparsity_training_test PRIVATE Autograd)

enable_testing()
add_test(NAME sparsity_training_test COMMAND sparsity_training_test)
