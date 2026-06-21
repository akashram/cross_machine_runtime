# gpu_engine/hopper — Steps 18–19: Hopper TMA + WGMMA
#
# gpu_hopper: INTERFACE library exposing tma.cuh and wgmma.cuh.
#   tma.cuh: CUtensorMap descriptor creation (driver API) + async tile load device code.
#   wgmma.cuh: PTX-level wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 wrapper.
#   Both files are fully guarded behind __CUDA_ARCH__ >= 900; they compile
#   cleanly on any GPU but emit no device code below sm_90.
#
# hopper_bench: TMA bandwidth vs cudaMemcpyAsync, WGMMA throughput vs cuBLAS.
#   The bench is also guarded: on non-H100 hardware it prints a message and exits.
#
# Required instance: p5.48xlarge (H100 SXM, CC 9.0) and CUDA 12.0+.
# Compile with -arch=sm_90a to enable TMA + WGMMA instruction set.

find_package(CUDAToolkit REQUIRED)

add_library(gpu_hopper INTERFACE)
target_include_directories(gpu_hopper INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_hopper INTERFACE CUDA::cudart CUDA::cuda_driver)

if(CMAKE_CUDA_ARCHITECTURES MATCHES "90")
    # Only emit sm_90a code when explicitly targeting H100
    add_executable(hopper_bench hopper_bench.cu)
    target_apply_cuda_flags(hopper_bench)
    target_compile_options(hopper_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90a>)
    target_link_libraries(hopper_bench PRIVATE CUDA::cudart CUDA::cublas CUDA::cuda_driver gpu_hopper)
    set_target_properties(hopper_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
else()
    # On non-H100 builds, still build the binary but it will detect at runtime
    add_executable(hopper_bench hopper_bench.cu)
    target_apply_cuda_flags(hopper_bench)
    target_link_libraries(hopper_bench PRIVATE CUDA::cudart CUDA::cublas CUDA::cuda_driver gpu_hopper)
    set_target_properties(hopper_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()
