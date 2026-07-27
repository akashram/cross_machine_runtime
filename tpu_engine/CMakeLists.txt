cmake_minimum_required(VERSION 3.25)

# Phase 8: TPU Backend
# Mostly Python/JAX-based — no CMake targets for those steps (run directly
# with `python3 step_dir/script.py` on a TPU VM).
#
# stablehlo_lower (step 3) is the exception: it's a C++ MLIR pass, built on
# top of compiler/dialect (Phase 4), so it needs both MLIR_DIR (same gate as
# compiler/) and STABLEHLO_DIR (github.com/openxla/stablehlo, built against
# the same LLVM commit — not part of upstream LLVM/MLIR itself).
if(DEFINED MLIR_DIR AND DEFINED STABLEHLO_DIR)
  add_subdirectory(stablehlo_lower)
else()
  message(STATUS "MLIR_DIR/STABLEHLO_DIR not set — tpu_engine/stablehlo_lower stub only (skipped)")
endif()

message(STATUS "tpu_engine/ is otherwise Python/JAX-based — no CMake targets")
