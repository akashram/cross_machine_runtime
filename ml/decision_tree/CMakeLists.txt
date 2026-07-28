# Step 1: CART decision tree — portable, builds and runs everywhere
# including this Mac.

add_library(DecisionTree decision_tree.cpp)
target_include_directories(DecisionTree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(DecisionTree PUBLIC cxx_std_23)
set_project_warnings(DecisionTree)

add_executable(decision_tree_test decision_tree_test.cpp)
target_link_libraries(decision_tree_test PRIVATE DecisionTree)
set_project_warnings(decision_tree_test)

enable_testing()
add_test(NAME decision_tree_test COMMAND decision_tree_test)
