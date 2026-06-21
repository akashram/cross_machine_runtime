# gpu_engine/p2p — Step 14: GPUDirect P2P bandwidth
#
# gpu_p2p: INTERFACE library exposing p2p.h (header-only, host code).
#   Wraps cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess /
#   cudaMemcpyPeerAsync with a bandwidth measurement API.
#
# p2p_bench: prints peer-access matrix, then sweeps transfer sizes for each
#   GPU pair and compares direct P2P vs host-staged bandwidth.
#   Falls back to intra-device memcpy when only one GPU is present.
#
# Recommended instance: p4d.24xlarge (8× A100 + NVLink-3) for NVLink numbers.
# g4dn.12xlarge (4× T4 + PCIe) for PCIe baseline.

add_library(gpu_p2p INTERFACE)
target_include_directories(gpu_p2p INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(gpu_p2p INTERFACE CUDA::cudart)

add_executable(p2p_bench p2p_bench.cu)
target_apply_cuda_flags(p2p_bench)
target_link_libraries(p2p_bench PRIVATE CUDA::cudart gpu_p2p)
set_target_properties(p2p_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
