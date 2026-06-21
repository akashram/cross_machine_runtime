# gpu_engine/precision — Steps 15–17: mixed precision + Tensor Core alignment
#
# gpu_precision: INTERFACE library exposing mixed_precision.h and tensor_align.h.
#   mixed_precision.h: LossScaler (host), cast_fp32_to_bf16/bf16_to_fp32 (device),
#     check_overflow (device).  Requires sm_80+ for __bfloat16 Tensor Cores.
#   tensor_align.h: alignment constants per DType, check helper, alignment cliff
#     sweep using cuBLAS.
#
# precision_bench:
#   - FP32 (TF32) vs BF16 GEMM throughput comparison
#   - LossScaler host-side simulation
#   - Alignment cliff sweep M=1..40, N=K=4096 (TF32 alignment = 8)
#
# Requires: A100 (sm_80) for BF16 Tensor Cores.  H100 (sm_90) for FP8.
# On older GPUs the BF16 path is guarded at runtime; only FP32 results shown.

find_package(CUDAToolkit REQUIRED)

add_library(gpu_precision INTERFACE)
target_include_directories(gpu_precision INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_precision INTERFACE CUDA::cudart CUDA::cublas)

add_executable(precision_bench precision_bench.cu)
target_apply_cuda_flags(precision_bench)
target_link_libraries(precision_bench PRIVATE CUDA::cudart CUDA::cublas gpu_precision)
set_target_properties(precision_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
