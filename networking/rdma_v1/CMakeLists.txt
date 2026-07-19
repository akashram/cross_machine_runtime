# Step 2: RDMA transport v1. Two independent targets:
#   tcp_baseline_bench — portable (POSIX sockets + threads), builds
#     everywhere including this Mac; see README.md for captured output.
#   rdma_transport — needs Linux + libfabric + an EFA NIC; only built if
#     both are present (this file is only reached on Linux at all, per
#     the gate in networking/CMakeLists.txt, but libfabric itself is a
#     separate, optional dependency even there).

add_executable(tcp_baseline_bench tcp_baseline_bench.cpp)
find_package(Threads REQUIRED)
target_link_libraries(tcp_baseline_bench PRIVATE Threads::Threads)

enable_testing()
add_test(NAME tcp_baseline_bench_loopback COMMAND tcp_baseline_bench loopback)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_library(LIBFABRIC_LIB fabric)
  if(LIBFABRIC_LIB)
    add_library(RdmaTransportV1 rdma_transport.cpp)
    target_link_libraries(RdmaTransportV1 PUBLIC ${LIBFABRIC_LIB})
  else()
    message(STATUS "rdma_v1: libfabric not found — rdma_transport skipped, tcp_baseline_bench still builds")
  endif()
endif()
