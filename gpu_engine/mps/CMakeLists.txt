# gpu_engine/mps — Step 22: CUDA Multi-Process Service (MPS) benchmark
#
# mps_bench: forks N child processes, each submitting K small CUDA kernels.
#   Measures wall time and per-kernel latency with and without MPS.
#   MPS is Linux-only; the bench still builds and provides a baseline on Linux.
#
# To test:
#   Without MPS: ./mps_bench [n_procs] [n_kernels]
#   With MPS:    sudo bash setup_mps.sh && ./mps_bench [n_procs] [n_kernels]
#                then sudo nvidia-smi -i 0 -c DEFAULT to restore

find_package(CUDAToolkit REQUIRED)

if(UNIX AND NOT APPLE)
    add_executable(mps_bench mps_bench.cu)
    target_apply_cuda_flags(mps_bench)
    target_link_libraries(mps_bench PRIVATE CUDA::cudart)
    set_target_properties(mps_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
else()
    message(STATUS "gpu_engine/mps: MPS requires Linux; skipping mps_bench on this platform")
endif()
