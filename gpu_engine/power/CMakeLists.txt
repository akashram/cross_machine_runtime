# gpu_engine/power — Step 23: NVML GPU power monitoring
#
# gpu_power: INTERFACE library exposing power_monitor.h.
#   Include with POWER_MONITOR_IMPL defined in exactly one .cu file to get
#   the full NVML implementation.  Other TUs get stub no-ops.
#
# power_bench: monitors idle / STREAM TRIAD / SGEMM / HGEMM workloads.
#   Prints avg power, peak power, energy (J), max temp, and throttle status.
#   Requires: Linux + NVIDIA driver + NVML (libnvidia-ml.so).

find_package(CUDAToolkit REQUIRED)

# NVML is part of the CUDA Toolkit on Linux
find_library(NVML_LIB nvidia-ml
             HINTS "${CUDAToolkit_ROOT}/lib64"
                   "${CUDAToolkit_ROOT}/lib"
                   "/usr/lib/x86_64-linux-gnu"
                   "/usr/lib64")

add_library(gpu_power INTERFACE)
target_include_directories(gpu_power INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_power INTERFACE CUDA::cudart)

if(NVML_LIB AND UNIX AND NOT APPLE)
    message(STATUS "Found NVML: ${NVML_LIB}")
    add_executable(power_bench power_bench.cu)
    target_apply_cuda_flags(power_bench)
    target_compile_definitions(power_bench PRIVATE POWER_MONITOR_IMPL)
    target_link_libraries(power_bench PRIVATE
        gpu_power CUDA::cudart CUDA::cublas ${NVML_LIB})
    set_target_properties(power_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
else()
    if(NOT NVML_LIB)
        message(STATUS "gpu_engine/power: NVML not found; skipping power_bench")
    else()
        message(STATUS "gpu_engine/power: skipping power_bench (requires Linux)")
    endif()
endif()
