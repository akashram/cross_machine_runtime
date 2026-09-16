# Phase 21 step 3: literature/vendor-doc-grounded parallel storage system
# comparison. Header-only, no dependencies.

add_executable(storage_comparison_test storage_comparison_test.cpp)
target_include_directories(storage_comparison_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_project_warnings(storage_comparison_test)

enable_testing()
add_test(NAME storage_comparison_test COMMAND storage_comparison_test)
