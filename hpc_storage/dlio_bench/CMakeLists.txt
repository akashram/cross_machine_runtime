# Phase 21 step 9: DLIO-style AI I/O benchmark. Real DataLoader (data_loading)
# + real transformer/ forward-pass compute.

add_executable(dlio_bench_test dlio_bench_test.cpp)
target_link_libraries(dlio_bench_test PRIVATE DataLoading Transformer)
set_project_warnings(dlio_bench_test)

enable_testing()
add_test(NAME dlio_bench_test COMMAND dlio_bench_test)
