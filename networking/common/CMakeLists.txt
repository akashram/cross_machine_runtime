# Shared substrate for Phase 5 steps 11-23 (and rdma_v1's TCP baseline).
# Pure POSIX sockets + threads — no Linux-only API, so unlike most of
# networking/ this builds and runs everywhere, including this Mac. Added
# unconditionally from the root CMakeLists.txt, ahead of the Linux-only
# gate the rest of networking/ sits behind.

find_package(Threads REQUIRED)

add_library(NetCommon channel.cpp)
target_include_directories(NetCommon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(NetCommon PUBLIC cxx_std_20)
target_link_libraries(NetCommon PUBLIC Threads::Threads)

add_executable(channel_test channel_test.cpp)
target_link_libraries(channel_test PRIVATE NetCommon)

enable_testing()
add_test(NAME channel_test COMMAND channel_test)
