# Step 14: 1F1B pipeline schedule. Portable: pure C++ discrete-event
# simulation, no CUDA/Linux dependency.

add_executable(pipeline_1f1b_test pipeline_1f1b_test.cpp)
target_include_directories(pipeline_1f1b_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_test(NAME pipeline_1f1b_test COMMAND pipeline_1f1b_test)
