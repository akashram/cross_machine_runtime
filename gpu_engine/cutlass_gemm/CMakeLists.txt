# gpu_engine/cutlass_gemm — Step 26: CUTLASS GEMM
#
# cutlass_gemm_bench: instantiates a CUTLASS SM80 TensorOp GEMM
# (cutlass::gemm::device::Gemm), benchmarks vs. gpu_engine/kernels' hand-
# written tiled/WMMA GEMM and gpu_engine/hopper's hand-written WGMMA kernel.
#
# CUTLASS is header-only (no library to build/link), so it is pulled via
# FetchContent rather than find_package — this is the first FetchContent
# use in the repo; every other external dependency so far (gRPC,
# FlatBuffers, libfabric, libbpf) is either a system package or vendored,
# see networking/CMakeLists.txt. FetchContent is the right tool here
# specifically because CUTLASS ships no distro package and is header-only,
# so there is nothing to find_package() against.
#
# Required instance: any CUDA GPU with compute capability 8.0+ (A100-class,
# p3.2xlarge/p4d.24xlarge — same as gpu_engine/kernels' WMMA path). This
# target is never built on Mac: the whole gpu_engine/ subtree is gated
# behind check_language(CUDA) at the root CMakeLists.txt.

include(FetchContent)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        v3.5.1
)
# CUTLASS_ENABLE_EXAMPLES/TESTS off — this repo only needs the headers.
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cutlass)

find_package(CUDAToolkit REQUIRED)

add_executable(cutlass_gemm_bench cutlass_gemm.cu)
target_apply_cuda_flags(cutlass_gemm_bench)
target_link_libraries(cutlass_gemm_bench PRIVATE
    CUDA::cudart CUDA::cublas
    nvidia::cutlass::cutlass
)
target_compile_options(cutlass_gemm_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>
)
set_target_properties(cutlass_gemm_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
