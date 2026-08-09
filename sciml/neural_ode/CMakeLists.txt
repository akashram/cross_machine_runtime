# Phase 18 step 4: Neural ODEs, adjoint-method gradients. Depends on step
# 1's SciMLOdeSolver (reuses rk4 directly for both the forward pass and
# the adjoint backward pass).

add_library(SciMLNeuralOde INTERFACE)
target_include_directories(SciMLNeuralOde INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SciMLNeuralOde INTERFACE SciMLOdeSolver)

add_executable(neural_ode_test neural_ode_test.cpp)
target_link_libraries(neural_ode_test PRIVATE SciMLNeuralOde)
set_project_warnings(neural_ode_test)

enable_testing()
add_test(NAME neural_ode_test COMMAND neural_ode_test)
