# Step 7: transferability study. Portable, no CUDA/Linux dependency.

add_executable(transferability_test transferability_test.cpp)
target_link_libraries(transferability_test PRIVATE AdversarialPGD)
set_project_warnings(transferability_test)

enable_testing()
add_test(NAME transferability_test COMMAND transferability_test)
