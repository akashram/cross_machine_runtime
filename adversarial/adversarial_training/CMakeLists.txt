# Step 5: adversarial training defense. Portable, header-only, no
# CUDA/Linux dependency.

add_library(AdversarialTraining INTERFACE)
target_include_directories(AdversarialTraining INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AdversarialTraining INTERFACE AdversarialPGD)

add_executable(adversarial_training_test adversarial_training_test.cpp)
target_link_libraries(adversarial_training_test PRIVATE AdversarialTraining)
set_project_warnings(adversarial_training_test)

enable_testing()
add_test(NAME adversarial_training_test COMMAND adversarial_training_test)
