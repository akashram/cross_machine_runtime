# Phase 22 step 5: rack power/cooling capacity model. Header-only, no
# dependencies.

add_library(RackPowerModel INTERFACE)
target_include_directories(RackPowerModel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(rack_power_model_test rack_power_model_test.cpp)
target_link_libraries(rack_power_model_test PRIVATE RackPowerModel)
set_project_warnings(rack_power_model_test)

add_test(NAME rack_power_model_test COMMAND rack_power_model_test)
