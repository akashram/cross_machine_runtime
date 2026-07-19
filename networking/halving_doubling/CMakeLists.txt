# Step 12: Recursive halving-doubling all-reduce — portable, builds and
# runs everywhere including this Mac. Falls back to RingAllreduce for
# non-power-of-2 world sizes (see halving_doubling.h).

add_library(HalvingDoubling halving_doubling.cpp)
target_link_libraries(HalvingDoubling PUBLIC NetCommon RingAllreduce)
target_include_directories(HalvingDoubling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(halving_doubling_test halving_doubling_test.cpp)
target_link_libraries(halving_doubling_test PRIVATE HalvingDoubling)

enable_testing()
add_test(NAME halving_doubling_test COMMAND halving_doubling_test)
