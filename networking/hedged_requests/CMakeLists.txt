# Step 22: Hedged requests — portable, builds and runs everywhere
# including this Mac.

add_library(HedgedRequests hedged_request.cpp)
target_include_directories(HedgedRequests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Threads REQUIRED)
target_link_libraries(HedgedRequests PUBLIC Threads::Threads)

add_executable(hedged_request_test hedged_request_test.cpp)
target_link_libraries(hedged_request_test PRIVATE HedgedRequests)

enable_testing()
add_test(NAME hedged_request_test COMMAND hedged_request_test)
