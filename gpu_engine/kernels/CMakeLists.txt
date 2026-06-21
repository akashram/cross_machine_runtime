# gpu_engine/kernels — Steps 8–9: element-wise kernels + GEMM
#
# gpu_kernels: INTERFACE library exposing elementwise.cuh and gemm.cuh.
#   Both are header-only device code (no .cu translation unit).
#
# elementwise_bench: add/mul/relu/gelu/softmax throughput (GB/s vs. HBM peak).
#
# gemm_bench: naive / tiled<16> / tiled<32> / WMMA / cuBLAS comparison (TFLOPS).
#   WMMA path compiled only when CMAKE_CUDA_ARCHITECTURES includes Volta+ (sm_70+).

find_package(CUDAToolkit REQUIRED)

add_library(gpu_kernels INTERFACE)
target_include_directories(gpu_kernels INTERFACE ${CMAKE_SOURCE_DIR})

add_executable(elementwise_bench elementwise_bench.cu)
target_apply_cuda_flags(elementwise_bench)
target_link_libraries(elementwise_bench PRIVATE CUDA::cudart gpu_kernels)
set_target_properties(elementwise_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

add_executable(gemm_bench gemm_bench.cu)
target_apply_cuda_flags(gemm_bench)
target_link_libraries(gemm_bench PRIVATE CUDA::cudart CUDA::cublas gpu_kernels)
set_target_properties(gemm_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
