# gpu_engine/streams — Step 3: Stream manager
#
# gpu_streams: INTERFACE library exposing stream_pool.h and event_pool.h.
# Header-only.
#
# streams_bench: measures cudaStreamCreate/cudaEventCreate latency vs pool
# acquire, then sequential H2D+kernel vs overlapped H2D‖kernel.

add_library(gpu_streams INTERFACE)
target_include_directories(gpu_streams INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/gpu_engine
)
target_link_libraries(gpu_streams INTERFACE CUDA::cudart)

add_executable(streams_bench streams_bench.cu)
target_apply_cuda_flags(streams_bench)
target_link_libraries(streams_bench PRIVATE CUDA::cudart gpu_streams gpu_memory)
set_target_properties(streams_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
