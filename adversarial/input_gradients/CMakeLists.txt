# Step 1: input gradients via the existing generic autograd tape. Portable,
# header-only, no CUDA/Linux dependency.

add_library(AdversarialInputGrad INTERFACE)
target_include_directories(AdversarialInputGrad INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AdversarialInputGrad INTERFACE AdversarialTask)

add_executable(input_gradients_test input_gradients_test.cpp)
target_link_libraries(input_gradients_test PRIVATE AdversarialInputGrad)
set_project_warnings(input_gradients_test)

enable_testing()
add_test(NAME input_gradients_test COMMAND input_gradients_test)
