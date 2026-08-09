# Phase 17 step 8: analog circuit transient surrogate (RC step response).
# Header-only, no dependencies.

add_library(AnalogCircuitTransient INTERFACE)
target_include_directories(AnalogCircuitTransient INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(circuit_transient_test circuit_transient_test.cpp)
target_link_libraries(circuit_transient_test PRIVATE AnalogCircuitTransient)
set_project_warnings(circuit_transient_test)

enable_testing()
add_test(NAME circuit_transient_test COMMAND circuit_transient_test)
