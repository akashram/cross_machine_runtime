# Step 3: HNSW (Malkov & Yashunin 2016), benchmarked directly against
# ml/knn's BallTree. Portable, no CUDA/Linux dependency.

add_library(HNSW hnsw.cpp)
target_include_directories(HNSW PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(HNSW PUBLIC KNN)
target_compile_features(HNSW PUBLIC cxx_std_23)
set_project_warnings(HNSW)

add_executable(hnsw_test hnsw_test.cpp)
target_link_libraries(hnsw_test PRIVATE HNSW)
set_project_warnings(hnsw_test)

enable_testing()
add_test(NAME hnsw_test COMMAND hnsw_test)
