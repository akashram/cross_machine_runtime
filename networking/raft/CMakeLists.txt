# Step 19: Raft consensus — portable, builds and runs everywhere
# including this Mac.

add_library(Raft raft.cpp)
target_link_libraries(Raft PUBLIC NetCommon)
target_include_directories(Raft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(raft_test raft_test.cpp)
target_link_libraries(raft_test PRIVATE Raft)

enable_testing()
add_test(NAME raft_test COMMAND raft_test)
