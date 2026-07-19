cmake_minimum_required(VERSION 3.25)

# Phase 5: Distributed Layer + Networking
#
# Split in two: algorithm steps (11-23) are written against the portable
# networking/common/ Channel abstraction (POSIX sockets — see
# networking/DESIGN.md) and build/run everywhere, including this Mac.
# Everything that needs a real EFA NIC, libfabric, AF_XDP, or ptp4l is
# Linux-only and gated below. TLA+ specs (tla_raft, tla_collective) have
# no build target either way — they're just checked in.

add_subdirectory(common)          # shared Channel/TcpChannel substrate
add_subdirectory(rdma_v1)         # Step 2 (tcp_baseline_bench half is portable; libfabric half self-gates)
add_subdirectory(efa_srd)         # Step 4 (pure decision logic, no libfabric needed)
add_subdirectory(ring_allreduce)  # Step 11
add_subdirectory(halving_doubling) # Step 12
add_subdirectory(tree_allreduce)  # Step 13
add_subdirectory(collectives)     # Step 14
add_subdirectory(topo_scheduler)  # Step 16 (portable scheduling algorithm)
add_subdirectory(vector_clocks)   # Step 17
add_subdirectory(chandy_lamport)  # Step 18
add_subdirectory(raft)            # Step 19
add_subdirectory(tla_raft)        # Step 20 (spec only, no build target)
add_subdirectory(backpressure)    # Step 21
add_subdirectory(hedged_requests) # Step 22
add_subdirectory(multitenancy)    # Step 23
add_subdirectory(tla_collective)  # Step 25 (spec only, no build target)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "networking/: EFA/AF_XDP/PTP/gRPC/chaos steps require Linux — skipped on ${CMAKE_SYSTEM_NAME}")
  return()
endif()

# Check for libfabric (required for EFA/RDMA steps)
find_library(LIBFABRIC_LIB fabric)
if(NOT LIBFABRIC_LIB)
  message(STATUS "libfabric not found — EFA steps build what they can without it")
endif()

add_subdirectory(efa_setup)       # Step 1
add_subdirectory(rdma_onesided)   # Step 3
add_subdirectory(ptp)             # Step 5
add_subdirectory(grpc_control)    # Step 6
add_subdirectory(flatbuffers_data) # Step 7
add_subdirectory(af_xdp)          # Step 8
add_subdirectory(userspace_net)   # Step 9
add_subdirectory(nic_deep_dive)   # Step 10
add_subdirectory(nccl_tuning)     # Step 15
add_subdirectory(chaos)           # Step 24
