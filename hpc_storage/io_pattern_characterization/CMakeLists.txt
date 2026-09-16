# Phase 21 step 1: I/O pattern characterization. Links against the real
# distributed_training/data_loading library target; checkpoint's
# sharded_checkpoint.h is header-only, included directly.

add_executable(io_pattern_test io_pattern_test.cpp)
target_include_directories(io_pattern_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/distributed_training/checkpoint
)
target_link_libraries(io_pattern_test PRIVATE DataLoading)
set_project_warnings(io_pattern_test)

enable_testing()
add_test(NAME io_pattern_test COMMAND io_pattern_test)
