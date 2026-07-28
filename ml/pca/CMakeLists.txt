# Step 7: PCA via randomized SVD (Halko, Martinsson & Tropp 2011).
# Portable, builds and runs everywhere including this Mac.

add_library(PCA pca.cpp)
target_include_directories(PCA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(PCA PUBLIC cxx_std_23)
set_project_warnings(PCA)

add_executable(pca_test pca_test.cpp)
target_link_libraries(pca_test PRIVATE PCA)
set_project_warnings(pca_test)

enable_testing()
add_test(NAME pca_test COMMAND pca_test)
