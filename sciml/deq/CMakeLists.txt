# Phase 18 step 5: Deep Equilibrium Models, implicit-function-theorem
# gradients. Depends on step 1's SciMLOdeSolver (reuses detail::solve_linear
# for the implicit backward pass).

add_library(SciMLDeq INTERFACE)
target_include_directories(SciMLDeq INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SciMLDeq INTERFACE SciMLOdeSolver)

add_executable(deq_test deq_test.cpp)
target_link_libraries(deq_test PRIVATE SciMLDeq)
set_project_warnings(deq_test)

enable_testing()
add_test(NAME deq_test COMMAND deq_test)
