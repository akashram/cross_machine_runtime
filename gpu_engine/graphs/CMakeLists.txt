# gpu_engine/graphs — Step 13: CUDA Graphs
#
# gpu_graphs: INTERFACE library exposing graph_runner.h (header-only, host code).
#   GraphRunner wraps cudaStreamBeginCapture / cudaStreamEndCapture /
#   cudaGraphInstantiate / cudaGraphLaunch and measures CPU launch overhead.
#
# graph_bench: chains CHAIN_LEN=20 simple kernels, compares eager vs graph
#   CPU submission overhead and GPU wall time.

add_library(gpu_graphs INTERFACE)
target_include_directories(gpu_graphs INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_graphs INTERFACE CUDA::cudart)

add_executable(graph_bench graph_bench.cu)
target_apply_cuda_flags(graph_bench)
target_link_libraries(graph_bench PRIVATE CUDA::cudart gpu_graphs)
set_target_properties(graph_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
