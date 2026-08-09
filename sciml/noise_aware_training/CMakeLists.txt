# Phase 18 step 10 (last step): noise-aware training through Phase 17
# step 1's real device model. Depends on step 9's SciMLMupScaling and
# Phase 17's AnalogDeviceModel.

add_library(SciMLNoiseAwareTraining INTERFACE)
target_include_directories(SciMLNoiseAwareTraining INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SciMLNoiseAwareTraining INTERFACE SciMLMupScaling AnalogDeviceModel)

add_executable(noise_aware_training_test noise_aware_training_test.cpp)
target_link_libraries(noise_aware_training_test PRIVATE SciMLNoiseAwareTraining)
set_project_warnings(noise_aware_training_test)

enable_testing()
add_test(NAME noise_aware_training_test COMMAND noise_aware_training_test)
