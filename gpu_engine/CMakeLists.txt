# Phase 3: GPU Backend
# Requires CUDA toolkit. Skipped entirely when CMAKE_CUDA_COMPILER is not set
# (e.g. macOS — NVIDIA dropped macOS CUDA support after 10.x).
# The root CMakeLists.txt gates this via check_language(CUDA) before
# calling add_subdirectory(gpu_engine).

cmake_minimum_required(VERSION 3.25)

find_package(CUDAToolkit REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake)

message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_subdirectory(device_query)

# Remaining subdirectories added as steps are implemented:
add_subdirectory(memory)          # Step 2
add_subdirectory(streams)         # Step 3
add_subdirectory(warp_primitives)  # Step 4
add_subdirectory(shared_mem)      # Step 5
# add_subdirectory(coalescing)    # Step 6
# add_subdirectory(occupancy)     # Step 7
# add_subdirectory(kernels)       # Steps 8-9
# add_subdirectory(ptx_sass)      # Step 10
# add_subdirectory(flash_attn)    # Steps 11-12
# add_subdirectory(graphs)        # Step 13
# add_subdirectory(p2p)           # Step 14
# add_subdirectory(precision)     # Steps 15-17
# add_subdirectory(hopper)        # Steps 18-19
# add_subdirectory(sparsity)      # Step 20
# add_subdirectory(roofline)      # Step 21
# add_subdirectory(mps)           # Step 22
# add_subdirectory(power)         # Step 23
# add_subdirectory(ci)            # Step 24
