# Phase 20 step 6: QAOA. Header-only, depends only on step 1.

add_library(QuantumQAOA INTERFACE)
target_include_directories(QuantumQAOA INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(QuantumQAOA INTERFACE QuantumStateVector)

add_executable(qaoa_test qaoa_test.cpp)
target_link_libraries(qaoa_test PRIVATE QuantumQAOA)
set_project_warnings(qaoa_test)

add_test(NAME qaoa_test COMMAND qaoa_test)
