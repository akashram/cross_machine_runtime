# Phase 20 step 5: VQE. Header-only, depends only on step 1.

add_library(QuantumVQE INTERFACE)
target_include_directories(QuantumVQE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(QuantumVQE INTERFACE QuantumStateVector)

add_executable(vqe_test vqe_test.cpp)
target_link_libraries(vqe_test PRIVATE QuantumVQE)
set_project_warnings(vqe_test)

add_test(NAME vqe_test COMMAND vqe_test)
