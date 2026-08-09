# Phase 18 step 2: stability + stiffness analysis. Depends on step 1's
# SciMLOdeSolver.

add_library(SciMLStiffness INTERFACE)
target_include_directories(SciMLStiffness INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SciMLStiffness INTERFACE SciMLOdeSolver)

add_executable(stiffness_test stiffness_test.cpp)
target_link_libraries(stiffness_test PRIVATE SciMLStiffness)
set_project_warnings(stiffness_test)

enable_testing()
add_test(NAME stiffness_test COMMAND stiffness_test)
