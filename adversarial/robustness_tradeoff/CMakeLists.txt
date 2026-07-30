# Step 6: robustness/accuracy tradeoff curve. Pure composition of steps
# 4/5 -- no new attack or training logic. Portable, no CUDA/Linux
# dependency.

add_executable(robustness_tradeoff_test robustness_tradeoff_test.cpp)
target_link_libraries(robustness_tradeoff_test PRIVATE AdversarialTraining AdversarialVulnMeasurement)
set_project_warnings(robustness_tradeoff_test)

enable_testing()
add_test(NAME robustness_tradeoff_test COMMAND robustness_tradeoff_test)
