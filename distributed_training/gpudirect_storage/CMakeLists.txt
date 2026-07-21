# Step 2: GPUDirect Storage — real cuFile API usage (code-complete, matching
# Phase 3's convention for GPU code that hasn't run yet). Unlike step 1,
# there's no portable CPU-only subset: GDS's entire point is bypassing the
# CPU, so it needs CUDA (already gated by the root CMakeLists' check_language)
# AND libcufile AND a GDS-capable NVMe device to actually run.

if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "gpudirect_storage/: no CUDA compiler — skipped")
  return()
endif()

find_library(CUFILE_LIB cufile)
find_path(CUFILE_INCLUDE_DIR cufile.h)
if(NOT CUFILE_LIB OR NOT CUFILE_INCLUDE_DIR)
  message(STATUS "gpudirect_storage/: libcufile not found — skipped (needs CUDA toolkit's GDS component)")
  return()
endif()

find_package(CUDAToolkit REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake)

add_executable(gds_bench gds_bench.cu)
target_apply_cuda_flags(gds_bench)
target_include_directories(gds_bench PRIVATE ${CUFILE_INCLUDE_DIR})
target_link_libraries(gds_bench PRIVATE CUDA::cudart ${CUFILE_LIB})
