# Step 1: Distributed data loading — WebDataset shard codec, rank sharding,
# multi-worker prefetch queue. Pure C++ (no CUDA/Linux dependency), builds
# and runs everywhere including this Mac.

find_package(Threads REQUIRED)

add_library(DataLoading webdataset_shard.cpp data_loader.cpp)
target_include_directories(DataLoading PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(DataLoading PUBLIC Threads::Threads)

add_executable(data_loader_test data_loader_test.cpp)
target_link_libraries(data_loader_test PRIVATE DataLoading)

enable_testing()
add_test(NAME data_loader_test COMMAND data_loader_test)
