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

# Phase 4: Compiler/IR (MLIR) — requires MLIR_DIR to be set
if(DEFINED MLIR_DIR)
  add_subdirectory(compiler)
else()
  message(STATUS "MLIR_DIR not set — compiler/ stub only (skipped)")
endif()

# Phases 5-10, 12: stubs — each CMakeLists.txt guards itself on missing deps
add_subdirectory(networking)
add_subdirectory(distributed_training)
add_subdirectory(fpga_engine)
add_subdirectory(tpu_engine)
add_subdirectory(inference_serving)
add_subdirectory(observability)
add_subdirectory(ml)
