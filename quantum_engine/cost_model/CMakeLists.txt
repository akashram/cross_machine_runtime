# Phase 20 step 10: cloud QPU hardware landscape + cost model.
# Header-only, no dependencies (literature comparison, not a simulation).

add_library(QuantumQpuLandscape INTERFACE)
target_include_directories(QuantumQpuLandscape INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(qpu_landscape_test qpu_landscape_test.cpp)
target_link_libraries(qpu_landscape_test PRIVATE QuantumQpuLandscape)
set_project_warnings(qpu_landscape_test)

add_test(NAME qpu_landscape_test COMMAND qpu_landscape_test)
