# Phase 17 step 2: resistive crossbar analog MAC simulation. Depends on
# step 1's AnalogDeviceModel.

add_library(AnalogCrossbarMac INTERFACE)
target_include_directories(AnalogCrossbarMac INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AnalogCrossbarMac INTERFACE AnalogDeviceModel)

add_executable(crossbar_mac_test crossbar_mac_test.cpp)
target_link_libraries(crossbar_mac_test PRIVATE AnalogCrossbarMac)
set_project_warnings(crossbar_mac_test)

enable_testing()
add_test(NAME crossbar_mac_test COMMAND crossbar_mac_test)
