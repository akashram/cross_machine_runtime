# Step 8: linear models -- elastic-net SGD, L-BFGS, logistic regression.
# Portable, builds and runs everywhere including this Mac.

add_library(LinearModel linear_model.cpp)
target_include_directories(LinearModel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(LinearModel PUBLIC cxx_std_23)
set_project_warnings(LinearModel)

add_executable(linear_model_test linear_model_test.cpp)
target_link_libraries(linear_model_test PRIVATE LinearModel)
set_project_warnings(linear_model_test)

enable_testing()
add_test(NAME linear_model_test COMMAND linear_model_test)
