# Phase 20 step 4: minimal QEC (3-qubit bit-flip code). Header-only,
# depends on steps 1 and 3.

add_library(QuantumQEC INTERFACE)
target_include_directories(QuantumQEC INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(QuantumQEC INTERFACE QuantumStateVector QuantumNoiseModel)

add_executable(qec_test qec_test.cpp)
target_link_libraries(qec_test PRIVATE QuantumQEC)
set_project_warnings(qec_test)

add_test(NAME qec_test COMMAND qec_test)
