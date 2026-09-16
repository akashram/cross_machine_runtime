# Phase 20 step 2: canonical algorithms (Deutsch-Jozsa, Grover, QFT) on
# state_vector's simulator. Header-only, depends only on step 1.

add_library(QuantumAlgorithms INTERFACE)
target_include_directories(QuantumAlgorithms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(QuantumAlgorithms INTERFACE QuantumStateVector)

add_executable(algorithms_test algorithms_test.cpp)
target_link_libraries(algorithms_test PRIVATE QuantumAlgorithms)
set_project_warnings(algorithms_test)

add_test(NAME algorithms_test COMMAND algorithms_test)
