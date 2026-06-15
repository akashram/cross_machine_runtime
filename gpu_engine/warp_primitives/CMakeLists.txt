# gpu_engine/warp_primitives — Step 4: warp-level primitives
#
# gpu_warp: INTERFACE library exposing warp_ops.h (device-only, header-only).
#
# warp_bench: reduce (warp-shuffle vs shared-memory), inclusive scan
# (Kogge-Stone vs Hillis-Steele), and ballot compaction vs atomicAdd.

add_library(gpu_warp INTERFACE)
target_include_directories(gpu_warp INTERFACE ${CMAKE_SOURCE_DIR})

add_executable(warp_bench warp_bench.cu)
target_apply_cuda_flags(warp_bench)
target_link_libraries(warp_bench PRIVATE CUDA::cudart gpu_warp)
set_target_properties(warp_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
