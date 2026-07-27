# Step 2: continuous batching scheduler — portable, builds and runs
# everywhere including this Mac (same "bookkeeping, not device bytes"
# split as paged_kv).

add_library(ContinuousBatcher batcher.cpp)
target_include_directories(ContinuousBatcher PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ContinuousBatcher PUBLIC cxx_std_23)
set_project_warnings(ContinuousBatcher)

add_executable(batcher_test batcher_test.cpp)
target_link_libraries(batcher_test PRIVATE ContinuousBatcher)
set_project_warnings(batcher_test)

add_executable(continuous_batching_bench continuous_batching_bench.cpp)
target_link_libraries(continuous_batching_bench PRIVATE ContinuousBatcher)
set_project_warnings(continuous_batching_bench)

enable_testing()
add_test(NAME batcher_test COMMAND batcher_test)
