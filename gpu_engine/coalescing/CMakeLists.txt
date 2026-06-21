# gpu_engine/coalescing — Step 6: memory coalescing validator
#
# coalescing_validator.h: header-only C++ library (no CUDA headers).
#   Runs ncu as a subprocess, parses its CSV output, returns CoalescingReport
#   per kernel. Used by the CI gate to fail the build on < threshold coalescing.
#
# coalescing_test: CUDA binary with a perfectly-coalesced kernel and a
#   deliberately strided (uncoalesced) kernel. Feed to coalescing_check.sh to
#   verify the validator and to calibrate ideal_ratio for this GPU.
#
# coalescing_check.sh: bash + python3 script for CI. No CMake integration
#   needed — run it directly on any benchmark binary.

add_library(gpu_coalescing INTERFACE)
target_include_directories(gpu_coalescing INTERFACE ${CMAKE_SOURCE_DIR})

add_executable(coalescing_test coalescing_test.cu)
target_apply_cuda_flags(coalescing_test)
target_link_libraries(coalescing_test PRIVATE CUDA::cudart gpu_coalescing)
set_target_properties(coalescing_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
