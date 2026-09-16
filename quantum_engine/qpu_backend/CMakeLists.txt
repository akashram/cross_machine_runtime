# Phase 20 step 9: cross-machine runtime integration. Header-only for
# the quantum-specific half; links inference_serving's real ServingRouter
# library for the integration test (no changes to ServingRouter's own
# CMakeLists.txt needed -- it's already a portable, always-built target).

add_library(QuantumQPUBackend INTERFACE)
target_include_directories(QuantumQPUBackend INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(QuantumQPUBackend INTERFACE QuantumStateVector)

add_executable(qpu_backend_test qpu_backend_test.cpp)
target_link_libraries(qpu_backend_test PRIVATE QuantumQPUBackend ServingRouter)
set_project_warnings(qpu_backend_test)

add_test(NAME qpu_backend_test COMMAND qpu_backend_test)
