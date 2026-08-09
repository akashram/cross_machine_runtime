# Phase 17 step 6: systolic array design-space sweep. Depends on step 5's
# AnalogDataflowModel.

add_library(AnalogSystolicSweep INTERFACE)
target_include_directories(AnalogSystolicSweep INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AnalogSystolicSweep INTERFACE AnalogDataflowModel)

add_executable(systolic_sweep_test systolic_sweep_test.cpp)
target_link_libraries(systolic_sweep_test PRIVATE AnalogSystolicSweep)
set_project_warnings(systolic_sweep_test)

enable_testing()
add_test(NAME systolic_sweep_test COMMAND systolic_sweep_test)
