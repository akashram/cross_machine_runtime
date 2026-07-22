# Minimal decoder-only transformer + character-level tokenizer. Not part
# of PLAN.md's original 12 phases — added to give distributed_training/'s
# RLHF steps (22-25: SFT, reward model, PPO, DPO) a real model to train,
# instead of stubbing those steps or reducing "language model" to a toy
# single-step classifier. See PLAN.md's "Minimal Transformer" section and
# this directory's README.md for the full rationale and scope.
#
# Portable: builds on distributed_training/autograd (Matrix),
# distributed_training/tensor_parallel_attn (causal attention primitive),
# and distributed_training/seq_parallel (LayerNorm) — no CUDA/Linux
# dependency, no other transformer/ dependents build yet, so this needs no
# platform gate.

add_library(Transformer INTERFACE)
target_include_directories(Transformer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Transformer INTERFACE Autograd)

add_executable(transformer_test transformer_test.cpp)
target_link_libraries(transformer_test PRIVATE Transformer)

enable_testing()
add_test(NAME transformer_test COMMAND transformer_test)
