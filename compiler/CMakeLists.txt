cmake_minimum_required(VERSION 3.25)

# Phase 4: Compiler / IR (MLIR)
# Requires LLVM/MLIR built from source on Linux.
# Build LLVM first: see compiler/mlir_setup/build_llvm.sh
# Then configure with: cmake -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir ..

if(NOT DEFINED MLIR_DIR)
  message(STATUS "MLIR_DIR not set — compiler/ stub only (skipping build)")
  return()
endif()

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include(HandleLLVMOptions)

add_subdirectory(mlir_setup)       # Step 1: LLVM/MLIR build setup
add_subdirectory(dialect)          # Steps 2-3: dialect design + registration
add_subdirectory(shape_inference)  # Step 4: shape inference pass
add_subdirectory(fusion)           # Step 5: operator fusion pass
add_subdirectory(affine_lower)     # Step 6: Affine dialect lowering
add_subdirectory(mem_planning)     # Step 7: memory planning pass
add_subdirectory(remat)            # Step 8: rematerialization pass
add_subdirectory(placement)        # Step 9: device placement pass
add_subdirectory(sharding)         # Step 10: auto-sharding pass
add_subdirectory(kernel_spec)      # Step 11: kernel specialization
add_subdirectory(aot)              # Step 12: AOT compilation pipeline
add_subdirectory(cost_model)       # Step 13: device cost model
add_subdirectory(fuzzing)          # Step 14: libFuzzer for IR parser
add_subdirectory(upstream)         # Step 15: LLVM upstream tracking
