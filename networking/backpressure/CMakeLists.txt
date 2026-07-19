# Step 21: Backpressure + load shedding — portable, builds and runs
# everywhere including this Mac.

add_library(Backpressure backpressure.cpp)
target_link_libraries(Backpressure PUBLIC NetCommon)
target_include_directories(Backpressure PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(backpressure_test backpressure_test.cpp)
target_link_libraries(backpressure_test PRIVATE Backpressure)

enable_testing()
add_test(NAME backpressure_test COMMAND backpressure_test)
