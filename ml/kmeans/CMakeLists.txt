# Step 6: k-means++ (Arthur & Vassilvitskii 2007) + Lloyd's iteration.
# Portable, builds and runs everywhere including this Mac.

add_library(KMeans kmeans.cpp)
target_include_directories(KMeans PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(KMeans PUBLIC cxx_std_23)
set_project_warnings(KMeans)

add_executable(kmeans_test kmeans_test.cpp)
target_link_libraries(kmeans_test PRIVATE KMeans)
set_project_warnings(kmeans_test)

enable_testing()
add_test(NAME kmeans_test COMMAND kmeans_test)
