# Step 17: Vector clocks — portable, builds and runs everywhere including
# this Mac.

add_library(VectorClocks vector_clock.cpp)
target_include_directories(VectorClocks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(vector_clock_test vector_clock_test.cpp)
target_link_libraries(vector_clock_test PRIVATE VectorClocks)

enable_testing()
add_test(NAME vector_clock_test COMMAND vector_clock_test)
