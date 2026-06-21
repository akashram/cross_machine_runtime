# gpu_engine/occupancy — Step 7: occupancy tuner
#
# gpu_occupancy: INTERFACE library exposing occupancy.h (header-only, host code).
#   Wraps cudaOccupancyMaxActiveBlocksPerMultiprocessor and
#   cudaOccupancyMaxPotentialBlockSize with a sweep API and print helper.
#
# occupancy_bench: sweeps block sizes and shared memory amounts for three kernels
#   (light, smem-heavy, register-heavy) and prints theoretical occupancy tables.

add_library(gpu_occupancy INTERFACE)
target_include_directories(gpu_occupancy INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_occupancy INTERFACE CUDA::cudart)

add_executable(occupancy_bench occupancy_bench.cu)
target_apply_cuda_flags(occupancy_bench)
target_link_libraries(occupancy_bench PRIVATE CUDA::cudart gpu_occupancy)
set_target_properties(occupancy_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
