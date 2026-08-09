# Phase 17 step 3: NVM device-class tradeoff comparison. Header-only, no
# dependencies.

add_library(AnalogNvmComparison INTERFACE)
target_include_directories(AnalogNvmComparison INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(nvm_comparison_test nvm_comparison_test.cpp)
target_link_libraries(nvm_comparison_test PRIVATE AnalogNvmComparison)
set_project_warnings(nvm_comparison_test)

enable_testing()
add_test(NAME nvm_comparison_test COMMAND nvm_comparison_test)
