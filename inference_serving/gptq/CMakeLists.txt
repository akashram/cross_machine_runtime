# Step 6: GPTQ INT4 group quantization — portable, builds and runs
# everywhere including this Mac. Quantization algorithm is pure CPU math;
# dequantize() + Matrix::matmul() is a correctness-oriented dequant-then-
# multiply, not the fused INT4xFP16 kernel a production deployment would
# want (a separate, GPU-specific optimization — see README.md).

add_library(Gptq gptq.cpp)
target_include_directories(Gptq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Gptq PUBLIC Autograd)
target_compile_features(Gptq PUBLIC cxx_std_23)
set_project_warnings(Gptq)

add_executable(gptq_test gptq_test.cpp)
target_link_libraries(gptq_test PRIVATE Gptq Transformer)
set_project_warnings(gptq_test)

enable_testing()
add_test(NAME gptq_test COMMAND gptq_test)
