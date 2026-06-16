# gpu_engine/shared_mem — Step 5: shared memory primitives
#
# gpu_shared: INTERFACE library exposing shared_ops.h (device-only, header-only).
# Depends on gpu_warp (for warp::scan_inclusive / warp::reduce_sum_lane0).
#
# shared_mem_bench: transpose (naive / tiled / padded), block scan
# (shared_ops.h vs Hillis-Steele), explicit bank conflict micro-benchmark.

add_library(gpu_shared INTERFACE)
target_include_directories(gpu_shared INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_shared INTERFACE gpu_warp)

add_executable(shared_mem_bench shared_mem_bench.cu)
target_apply_cuda_flags(shared_mem_bench)
target_link_libraries(shared_mem_bench PRIVATE CUDA::cudart gpu_shared)
set_target_properties(shared_mem_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
