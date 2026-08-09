# Phase 17 step 5: minimal Timeloop/Accelergy-style PPA/dataflow model.
# Header-only, no dependencies.

add_library(AnalogDataflowModel INTERFACE)
target_include_directories(AnalogDataflowModel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(dataflow_model_test dataflow_model_test.cpp)
target_link_libraries(dataflow_model_test PRIVATE AnalogDataflowModel)
set_project_warnings(dataflow_model_test)

enable_testing()
add_test(NAME dataflow_model_test COMMAND dataflow_model_test)
