# gpu_engine/flash_attn — Steps 11–12: Flash Attention forward + backward
#
# gpu_flash_attn: INTERFACE library exposing flash_attn_fwd.cuh and
#   flash_attn_bwd.cuh (header-only device code).
#
# flash_attn_bench: compares naive O(N²) attention vs Flash Attention forward
#   (latency + HBM footprint), then times the backward pass.
#   cuDNN comparison is deferred — add CUDA::cudnn linkage and #ifdef guard
#   when cuDNN is available on the target instance.

add_library(gpu_flash_attn INTERFACE)
target_include_directories(gpu_flash_attn INTERFACE ${CMAKE_SOURCE_DIR})

add_executable(flash_attn_bench flash_attn_bench.cu)
target_apply_cuda_flags(flash_attn_bench)
target_link_libraries(flash_attn_bench PRIVATE CUDA::cudart gpu_flash_attn)
set_target_properties(flash_attn_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
