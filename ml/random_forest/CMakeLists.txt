# Step 2: Random Forest — bagging over DecisionTree (step 1), parallelized
# across trees with foundation/'s work-stealing pool. Portable, builds and
# runs everywhere including this Mac.

add_library(RandomForest random_forest.cpp)
target_include_directories(RandomForest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(RandomForest PRIVATE ${CMAKE_SOURCE_DIR}/foundation)
target_link_libraries(RandomForest PUBLIC DecisionTree)
target_compile_features(RandomForest PUBLIC cxx_std_23)
set_project_warnings(RandomForest)

add_executable(random_forest_test random_forest_test.cpp)
target_link_libraries(random_forest_test PRIVATE RandomForest)
set_project_warnings(random_forest_test)

enable_testing()
add_test(NAME random_forest_test COMMAND random_forest_test)
