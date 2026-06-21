# gpu_engine/roofline — Step 21: GPU Roofline model
#
# gpu_roofline: INTERFACE library exposing roofline.h.
#   - measure_peak_flops_tflops(): cuBLAS SGEMM/HGEMM at 4096³
#   - measure_peak_bandwidth_gbs(): STREAM TRIAD on 256 MB arrays
#   - classify_kernel(): maps achieved TFLOPS + AI to compute/bandwidth bound
#   - print_roofline_result(): formatted output
#
# roofline_bench: measures hardware ceilings and classifies representative
#   kernels (vec_add, SGEMM, HGEMM, softmax max-pass).

find_package(CUDAToolkit REQUIRED)

add_library(gpu_roofline INTERFACE)
target_include_directories(gpu_roofline INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_roofline INTERFACE CUDA::cudart CUDA::cublas)

add_executable(roofline_bench roofline_bench.cu)
target_apply_cuda_flags(roofline_bench)
target_link_libraries(roofline_bench PRIVATE
    gpu_roofline CUDA::cudart CUDA::cublas)
set_target_properties(roofline_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
