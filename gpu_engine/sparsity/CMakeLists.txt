# gpu_engine/sparsity — Step 20: 2:4 structured sparsity
#
# gpu_sparsity: INTERFACE library exposing sparse_24.h.
#   - prune_2_4 CUDA kernel: zeros smallest 2-of-4 abs values per group per row.
#   - compress_2_4_cpu: CPU reference encoder (values + 4-bit position metadata).
#   - run_sparse_matmul: cuSPARSELt path (guarded by CUSPARSELT_VERSION).
#
# sparsity_bench: prune→compress→sparse matmul vs dense cuBLAS FP16.
#   Required hardware: sm_80+ (A100 or H100). cuSPARSELt needed for sparse path.
#   Without cuSPARSELt, the dense pruning/verification path still runs.
#
# cuSPARSELt is NOT part of the CUDA Toolkit; install separately:
#   https://developer.nvidia.com/cusparselt-downloads
# Then set -DCUSPARSELT_PATH=/usr/local at cmake configure time.

find_package(CUDAToolkit REQUIRED)

add_library(gpu_sparsity INTERFACE)
target_include_directories(gpu_sparsity INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_sparsity INTERFACE CUDA::cudart)

# Optional cuSPARSELt linkage
set(CUSPARSELT_PATH "" CACHE PATH "Path to cuSPARSELt installation (optional)")
if(CUSPARSELT_PATH)
    find_library(CUSPARSELT_LIB cusparselt HINTS "${CUSPARSELT_PATH}/lib64"
                                                   "${CUSPARSELT_PATH}/lib")
    if(CUSPARSELT_LIB)
        message(STATUS "Found cuSPARSELt: ${CUSPARSELT_LIB}")
        target_compile_definitions(gpu_sparsity INTERFACE
            CUSPARSELT_VERSION=1
            CUSPARSELT_INCLUDE_PATH="${CUSPARSELT_PATH}/include")
        target_include_directories(gpu_sparsity INTERFACE "${CUSPARSELT_PATH}/include")
        target_link_libraries(gpu_sparsity INTERFACE ${CUSPARSELT_LIB})
    else()
        message(STATUS "cuSPARSELt not found at ${CUSPARSELT_PATH}; sparse matmul disabled")
    endif()
endif()

add_executable(sparsity_bench sparsity_bench.cu)
target_apply_cuda_flags(sparsity_bench)
target_link_libraries(sparsity_bench PRIVATE
    gpu_sparsity CUDA::cudart CUDA::cublas)
set_target_properties(sparsity_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
