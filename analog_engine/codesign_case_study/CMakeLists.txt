# Phase 17 step 7: hardware-algorithm co-design case study. Depends on
# step 1's AnalogDeviceModel, step 2's AnalogCrossbarMac (for
# quantize_level), inference_serving's real GptqQuantizer, and
# transformer/'s real trained model.

add_library(AnalogCodesignCaseStudy INTERFACE)
target_include_directories(AnalogCodesignCaseStudy INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AnalogCodesignCaseStudy INTERFACE AnalogDeviceModel AnalogCrossbarMac Gptq)

add_executable(codesign_case_study_test codesign_case_study_test.cpp)
target_link_libraries(codesign_case_study_test PRIVATE AnalogCodesignCaseStudy Transformer)
set_project_warnings(codesign_case_study_test)

enable_testing()
add_test(NAME codesign_case_study_test COMMAND codesign_case_study_test)
