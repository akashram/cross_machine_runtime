# Phase 20 step 1: hand-rolled state-vector simulator. Header-only, no
# dependencies.

add_library(QuantumStateVector INTERFACE)
target_include_directories(QuantumStateVector INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(state_vector_test state_vector_test.cpp)
target_link_libraries(state_vector_test PRIVATE QuantumStateVector)
set_project_warnings(state_vector_test)

add_test(NAME state_vector_test COMMAND state_vector_test)
