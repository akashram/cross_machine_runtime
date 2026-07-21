cmake_minimum_required(VERSION 3.25)

# Phase 6: Distributed GPU Training
# Requires multi-GPU instance (p4d.24xlarge) for most steps. Steps written
# against no GPU/Linux-only dependency (step 1's WebDataset loader so far)
# build and run everywhere, including this Mac, and are added before the
# platform gate below — see networking/CMakeLists.txt for the established
# pattern this follows.

add_subdirectory(data_loading) # Step 1 — portable, no CUDA/Linux dependency

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "distributed_training/: remaining steps require Linux+GPU — skipping on ${CMAKE_SYSTEM_NAME}")
  return()
endif()

add_subdirectory(gpudirect_storage)
add_subdirectory(data_parallel)
add_subdirectory(grad_accum)
add_subdirectory(grad_clipping)
add_subdirectory(autograd)
add_subdirectory(zero1)
add_subdirectory(zero2)
add_subdirectory(zero3)
add_subdirectory(zero_infinity)
add_subdirectory(col_row_linear)
add_subdirectory(tensor_parallel_attn)
add_subdirectory(seq_parallel)
add_subdirectory(pipeline_1f1b)
add_subdirectory(parallel_3d)
add_subdirectory(moe)
add_subdirectory(checkpoint)
add_subdirectory(compute_comm_overlap)
add_subdirectory(sync_batchnorm)
add_subdirectory(full_training_loop)
add_subdirectory(sparsity_training)
add_subdirectory(sft)
add_subdirectory(reward_model)
add_subdirectory(ppo_rlhf)
add_subdirectory(dpo)
