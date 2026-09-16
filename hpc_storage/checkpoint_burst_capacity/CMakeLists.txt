# Phase 21 step 6: checkpoint burst ("thundering herd") capacity model.
# Uses distributed_training/checkpoint's header-only sharded_checkpoint.h.

find_package(Threads REQUIRED)

add_executable(checkpoint_burst_capacity_test checkpoint_burst_capacity_test.cpp)
target_include_directories(checkpoint_burst_capacity_test PRIVATE
  ${CMAKE_SOURCE_DIR}/distributed_training/checkpoint
)
target_link_libraries(checkpoint_burst_capacity_test PRIVATE Threads::Threads)
set_project_warnings(checkpoint_burst_capacity_test)

enable_testing()
add_test(NAME checkpoint_burst_capacity_test COMMAND checkpoint_burst_capacity_test)
