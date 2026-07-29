# Step 5: k-NN via KD-tree (exact) and ball tree (approximate). Portable,
# builds and runs everywhere including this Mac.

add_library(KNN knn.cpp)
target_include_directories(KNN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(KNN PUBLIC cxx_std_23)
set_project_warnings(KNN)

add_executable(knn_test knn_test.cpp)
target_link_libraries(knn_test PRIVATE KNN)
set_project_warnings(knn_test)

# PLAN.md Phase 13 step 2: cosine-similarity ANN, a thin wrapper around
# BallTree above (composition, not a modified copy -- see cosine_ann.h).
add_library(CosineANN cosine_ann.cpp)
target_include_directories(CosineANN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(CosineANN PUBLIC KNN)
target_compile_features(CosineANN PUBLIC cxx_std_23)
set_project_warnings(CosineANN)

add_executable(cosine_ann_test cosine_ann_test.cpp)
target_link_libraries(cosine_ann_test PRIVATE CosineANN)
set_project_warnings(cosine_ann_test)

enable_testing()
add_test(NAME knn_test COMMAND knn_test)
add_test(NAME cosine_ann_test COMMAND cosine_ann_test)
