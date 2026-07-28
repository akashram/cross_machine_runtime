# Step 4: SVM via simplified SMO (Platt 1998). Portable, builds and runs
# everywhere including this Mac.

add_library(SVM svm.cpp)
target_include_directories(SVM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(SVM PUBLIC cxx_std_23)
set_project_warnings(SVM)

add_executable(svm_test svm_test.cpp)
target_link_libraries(svm_test PRIVATE SVM)
set_project_warnings(svm_test)

enable_testing()
add_test(NAME svm_test COMMAND svm_test)
