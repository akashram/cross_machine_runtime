# Step 5: speculative decoding — portable, builds and runs everywhere
# including this Mac. Uses transformer/'s real CPU-trained model as both
# draft and verifier, so this needs no GPU or hosted checkpoints.

add_library(SpecDecode spec_decode.cpp)
target_include_directories(SpecDecode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SpecDecode PUBLIC Transformer)
target_compile_features(SpecDecode PUBLIC cxx_std_23)
set_project_warnings(SpecDecode)

add_executable(spec_decode_test spec_decode_test.cpp)
target_link_libraries(spec_decode_test PRIVATE SpecDecode)
set_project_warnings(spec_decode_test)

enable_testing()
add_test(NAME spec_decode_test COMMAND spec_decode_test)
