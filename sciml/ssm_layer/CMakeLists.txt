# Phase 18 step 6: state-space model layer vs. self-attention. Header-only,
# no dependencies.

add_library(SciMLSsmLayer INTERFACE)
target_include_directories(SciMLSsmLayer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ssm_layer_test ssm_layer_test.cpp)
target_link_libraries(ssm_layer_test PRIVATE SciMLSsmLayer)
set_project_warnings(ssm_layer_test)

enable_testing()
add_test(NAME ssm_layer_test COMMAND ssm_layer_test)
