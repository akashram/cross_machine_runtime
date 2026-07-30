# Step 8 (stretch goal): randomized smoothing. Portable, header-only,
# no CUDA/Linux dependency.

add_library(AdversarialRandomizedSmoothing INTERFACE)
target_include_directories(AdversarialRandomizedSmoothing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AdversarialRandomizedSmoothing INTERFACE AdversarialPGD)

add_executable(randomized_smoothing_test randomized_smoothing_test.cpp)
target_link_libraries(randomized_smoothing_test PRIVATE AdversarialRandomizedSmoothing)
set_project_warnings(randomized_smoothing_test)

enable_testing()
add_test(NAME randomized_smoothing_test COMMAND randomized_smoothing_test)
