# Step 3: Gradient Boosted Trees (Friedman, Newton/second-order steps) --
# base learner is a regression tree fit to gradients/Hessians. Portable,
# builds and runs everywhere including this Mac.

add_library(GradientBoostedTrees gbt.cpp regression_tree.cpp)
target_include_directories(GradientBoostedTrees PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(GradientBoostedTrees PUBLIC DecisionTree)
target_compile_features(GradientBoostedTrees PUBLIC cxx_std_23)
set_project_warnings(GradientBoostedTrees)

add_executable(gbt_test gbt_test.cpp)
target_link_libraries(gbt_test PRIVATE GradientBoostedTrees)
set_project_warnings(gbt_test)

enable_testing()
add_test(NAME gbt_test COMMAND gbt_test)
