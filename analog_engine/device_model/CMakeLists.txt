# Phase 17 step 1: RRAM-like device non-ideality model. Header-only, no
# dependencies.

add_library(AnalogDeviceModel INTERFACE)
target_include_directories(AnalogDeviceModel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(device_model_test device_model_test.cpp)
target_link_libraries(device_model_test PRIVATE AnalogDeviceModel)
set_project_warnings(device_model_test)

enable_testing()
add_test(NAME device_model_test COMMAND device_model_test)
