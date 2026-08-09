# Phase 17 step 4: analog-vs-digital MAC energy model. Header-only, no
# dependencies.

add_library(AnalogEnergyModel INTERFACE)
target_include_directories(AnalogEnergyModel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(energy_model_test energy_model_test.cpp)
target_link_libraries(energy_model_test PRIVATE AnalogEnergyModel)
set_project_warnings(energy_model_test)

enable_testing()
add_test(NAME energy_model_test COMMAND energy_model_test)
