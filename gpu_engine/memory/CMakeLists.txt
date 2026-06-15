# gpu_engine/memory — Step 2: GPU memory management
#
# gpu_memory: INTERFACE library exposing gpu_alloc.h and tensor_cuda.h.
# Header-only; consumers link against CUDA::cudart directly.
#
# memory_bench: standalone benchmark — cudaMalloc latency,
# H2D/D2H (pageable vs pinned), D2D bandwidth.

add_library(gpu_memory INTERFACE)
target_include_directories(gpu_memory INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/gpu_engine
)
target_link_libraries(gpu_memory INTERFACE CUDA::cudart)

add_executable(memory_bench memory_bench.cu)
target_apply_cuda_flags(memory_bench)
target_link_libraries(memory_bench PRIVATE CUDA::cudart gpu_memory)
set_target_properties(memory_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
