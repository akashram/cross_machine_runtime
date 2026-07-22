cmake_minimum_required(VERSION 3.25)
project(cross_machine_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/Sanitizers.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Pgo.cmake)

enable_testing()

add_subdirectory(foundation)
add_subdirectory(cpu_engine)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_subdirectory(gpu_engine)
else()
  message(STATUS "CUDA not found — gpu_engine skipped")
endif()

# Phase 4: Compiler/IR (MLIR). cost_model has no MLIR dependency (pure
# C++ roofline arithmetic — see compiler/cost_model/README.md), so it
# builds and is testable everywhere, including this Mac. It must be added
# before the MLIR-gated compiler/ subdirectory below since PlacementPass
# and ShardingPass link against the CostModel target.
add_subdirectory(compiler/cost_model)

if(DEFINED MLIR_DIR)
  add_subdirectory(compiler)
else()
  message(STATUS "MLIR_DIR not set — compiler/ (except cost_model) stub only (skipped)")
endif()

# Phases 5-10, 12: stubs — each CMakeLists.txt guards itself on missing deps
add_subdirectory(networking)

# Minimal decoder-only transformer + tokenizer — not one of PLAN.md's
# original 12 phases; added to give distributed_training/'s RLHF steps
# (22-25) a real model. See transformer/README.md.
add_subdirectory(transformer)

add_subdirectory(distributed_training)
add_subdirectory(fpga_engine)
add_subdirectory(tpu_engine)
add_subdirectory(inference_serving)
add_subdirectory(observability)
add_subdirectory(ml)
