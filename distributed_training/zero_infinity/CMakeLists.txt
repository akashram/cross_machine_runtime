# Step 10: ZeRO-Infinity — CPU RAM / NVMe offload scheduling. Portable:
# the scheduling logic and analytical cost model need no CUDA/Linux
# dependency; real bandwidth numbers are a Results-table TODO (see README).

add_executable(offload_scheduler_test offload_scheduler_test.cpp)
target_include_directories(offload_scheduler_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_test(NAME offload_scheduler_test COMMAND offload_scheduler_test)
