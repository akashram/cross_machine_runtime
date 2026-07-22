# Step 17: Sharded checkpointing — real files, real background thread,
# real wall-clock overlap measurement. Portable: no CUDA/Linux dependency.

find_package(Threads REQUIRED)

add_executable(checkpoint_test checkpoint_test.cpp)
target_include_directories(checkpoint_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(checkpoint_test PRIVATE Threads::Threads)

enable_testing()
add_test(NAME checkpoint_test COMMAND checkpoint_test)
