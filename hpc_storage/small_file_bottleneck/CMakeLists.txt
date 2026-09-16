# Phase 21 step 2: small-file metadata bottleneck study. Links against the
# real DataLoading library target for tar_append/tar_finish/WebDatasetShardReader.

add_executable(small_file_bench small_file_bench.cpp)
target_link_libraries(small_file_bench PRIVATE DataLoading)
set_project_warnings(small_file_bench)

enable_testing()
add_test(NAME small_file_bench COMMAND small_file_bench)
