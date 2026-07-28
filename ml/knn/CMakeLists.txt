# Step 5: k-NN via KD-tree (exact) and ball tree (approximate). Portable,
# builds and runs everywhere including this Mac.

add_library(KNN knn.cpp)
target_include_directories(KNN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(KNN PUBLIC cxx_std_23)
set_project_warnings(KNN)

add_executable(knn_test knn_test.cpp)
target_link_libraries(knn_test PRIVATE KNN)
set_project_warnings(knn_test)

enable_testing()
add_test(NAME knn_test COMMAND knn_test)
