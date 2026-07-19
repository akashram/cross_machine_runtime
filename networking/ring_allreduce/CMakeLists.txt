# Step 11: Ring all-reduce — portable (built on networking/common::Channel),
# builds and runs everywhere including this Mac.

add_library(RingAllreduce ring_allreduce.cpp)
target_link_libraries(RingAllreduce PUBLIC NetCommon)
target_include_directories(RingAllreduce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ring_allreduce_test ring_allreduce_test.cpp)
target_link_libraries(ring_allreduce_test PRIVATE RingAllreduce)

enable_testing()
add_test(NAME ring_allreduce_test COMMAND ring_allreduce_test)
