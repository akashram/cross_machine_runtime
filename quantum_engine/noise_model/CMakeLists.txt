# Phase 20 step 3: noise/decoherence channel model. Header-only, depends
# only on step 1.

add_library(QuantumNoiseModel INTERFACE)
target_include_directories(QuantumNoiseModel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(QuantumNoiseModel INTERFACE QuantumStateVector)

add_executable(noise_model_test noise_model_test.cpp)
target_link_libraries(noise_model_test PRIVATE QuantumNoiseModel)
set_project_warnings(noise_model_test)

add_test(NAME noise_model_test COMMAND noise_model_test)
