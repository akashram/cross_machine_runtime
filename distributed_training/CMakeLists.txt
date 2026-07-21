cmake_minimum_required(VERSION 3.25)

# Phase 6: Distributed GPU Training
#
# Per CLAUDE.md's execution strategy, every step gets real code written and
# locally validated wherever the algorithm itself doesn't inherently require
# a GPU — the same "portable substrate first" split Phase 5's networking/
# used (see networking/CMakeLists.txt, networking/DESIGN.md). Concretely:
# distributed algorithms (data parallel, ZeRO, tensor/pipeline/3D
# parallelism, MoE, SyncBatchNorm, SFT/reward model/PPO/DPO, ...) are
# validated with a small CPU autograd + MLP (see autograd/) and simulated
# multi-rank communication over networking/common::Channel loopback threads
# — real gradients, real all-reduces, just no GPU underneath. Only steps
# that are inherently about bypassing the CPU or programming real GPU
# silicon (GPUDirect Storage, 2:4 sparse Tensor Core kernels) have no
# portable subset and stay hardware-gated below, matching Phase 3's
# convention for GPU code that hasn't run yet.
#
# All subdirectories are added unconditionally: steps not yet implemented
# have a one-line placeholder CMakeLists.txt that defines no targets, so
# add_subdirectory on them is a no-op until they're filled in.

add_subdirectory(data_loading)         # Step 1 — portable
add_subdirectory(data_parallel)        # Step 3 — portable
add_subdirectory(grad_accum)           # Step 4 — portable
add_subdirectory(grad_clipping)        # Step 5 — portable
add_subdirectory(autograd)             # Step 6 — portable
add_subdirectory(zero1)                # Step 7 — portable
add_subdirectory(zero2)                # Step 8 — portable
add_subdirectory(zero3)                # Step 9 — portable
add_subdirectory(zero_infinity)        # Step 10 — portable (CPU offload; NVMe offload leg is hardware-gated internally)
add_subdirectory(col_row_linear)       # Step 11 — portable
add_subdirectory(tensor_parallel_attn) # Step 12 — portable
add_subdirectory(seq_parallel)         # Step 13 — portable
add_subdirectory(pipeline_1f1b)        # Step 14 — portable
add_subdirectory(parallel_3d)          # Step 15 — portable
add_subdirectory(moe)                  # Step 16 — portable
add_subdirectory(checkpoint)           # Step 17 — portable
add_subdirectory(compute_comm_overlap) # Step 18 — portable
add_subdirectory(sync_batchnorm)       # Step 19 — portable
add_subdirectory(full_training_loop)   # Step 20 — portable
add_subdirectory(sparsity_training)    # Step 21 — portable (pruning logic); real 2:4 Tensor Core kernel is hardware-gated internally
add_subdirectory(sft)                  # Step 22 — portable (toy model)
add_subdirectory(reward_model)         # Step 23 — portable (toy model)
add_subdirectory(ppo_rlhf)             # Step 24 — portable (toy model)
add_subdirectory(dpo)                  # Step 25 — portable (toy model)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "distributed_training/: gpudirect_storage requires Linux+GPU — skipping on ${CMAKE_SYSTEM_NAME}")
  return()
endif()

add_subdirectory(gpudirect_storage) # Step 2 — hardware-gated, no portable subset (see its README.md)
