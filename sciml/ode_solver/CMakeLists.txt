# Phase 18 step 1: ODE solver library. Header-only, no dependencies.

add_library(SciMLOdeSolver INTERFACE)
target_include_directories(SciMLOdeSolver INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ode_solver_test ode_solver_test.cpp)
target_link_libraries(ode_solver_test PRIVATE SciMLOdeSolver)
set_project_warnings(ode_solver_test)

enable_testing()
add_test(NAME ode_solver_test COMMAND ode_solver_test)
