# Phase 18 step 9: muP output-layer LR-scaling study. Header-only, no
# dependencies (uses sciml::Vec from ssm_layer.h but not its trainer).

add_library(SciMLMupScaling INTERFACE)
target_include_directories(SciMLMupScaling INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SciMLMupScaling INTERFACE SciMLSsmLayer)

add_executable(mup_scaling_test mup_scaling_test.cpp)
target_link_libraries(mup_scaling_test PRIVATE SciMLMupScaling)
set_project_warnings(mup_scaling_test)

enable_testing()
add_test(NAME mup_scaling_test COMMAND mup_scaling_test)
