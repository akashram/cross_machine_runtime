# Step 7: INT8 KV cache quantization — portable, builds and runs
# everywhere including this Mac. Applies the quantization round-trip to
# transformer/'s real trained model's real K/V activations.

add_library(KvQuant kv_quant.cpp)
target_include_directories(KvQuant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(KvQuant PUBLIC Transformer)
target_compile_features(KvQuant PUBLIC cxx_std_23)
set_project_warnings(KvQuant)

add_executable(kv_quant_test kv_quant_test.cpp)
target_link_libraries(kv_quant_test PRIVATE KvQuant)
set_project_warnings(kv_quant_test)

enable_testing()
add_test(NAME kv_quant_test COMMAND kv_quant_test)
