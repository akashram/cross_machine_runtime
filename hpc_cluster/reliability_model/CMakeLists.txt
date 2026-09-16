# Phase 22 step 6: cluster reliability/MTBF-MTTR model. Header-only, no
# dependencies.

add_library(ReliabilityModel INTERFACE)
target_include_directories(ReliabilityModel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(reliability_model_test reliability_model_test.cpp)
target_link_libraries(reliability_model_test PRIVATE ReliabilityModel)
set_project_warnings(reliability_model_test)

add_test(NAME reliability_model_test COMMAND reliability_model_test)
