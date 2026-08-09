# Phase 18 step 3: SDE solvers (Euler-Maruyama, Milstein). Header-only, no
# dependencies.

add_library(SciMLSdeSolver INTERFACE)
target_include_directories(SciMLSdeSolver INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(sde_solver_test sde_solver_test.cpp)
target_link_libraries(sde_solver_test PRIVATE SciMLSdeSolver)
set_project_warnings(sde_solver_test)

enable_testing()
add_test(NAME sde_solver_test COMMAND sde_solver_test)
