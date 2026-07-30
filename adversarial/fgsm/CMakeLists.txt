# Step 2: FGSM. Portable, header-only, no CUDA/Linux dependency.

add_library(AdversarialFGSM INTERFACE)
target_include_directories(AdversarialFGSM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AdversarialFGSM INTERFACE AdversarialInputGrad)

add_executable(fgsm_test fgsm_test.cpp)
target_link_libraries(fgsm_test PRIVATE AdversarialFGSM)
set_project_warnings(fgsm_test)

enable_testing()
add_test(NAME fgsm_test COMMAND fgsm_test)
