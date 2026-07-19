# Step 14: Broadcast/reduce-scatter/all-gather collective library —
# portable, builds and runs everywhere including this Mac. Assembles
# ring_allreduce (step 11) and tree_allreduce (step 13) primitives rather
# than reimplementing them.

add_library(Collectives collectives.cpp)
target_link_libraries(Collectives PUBLIC NetCommon RingAllreduce TreeAllreduce)
target_include_directories(Collectives PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(collectives_test collectives_test.cpp)
target_link_libraries(collectives_test PRIVATE Collectives)

enable_testing()
add_test(NAME collectives_test COMMAND collectives_test)
