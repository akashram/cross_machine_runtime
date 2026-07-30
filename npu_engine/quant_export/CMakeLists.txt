# Phase 15 step 2: NPU quantization export pipeline. Portable — reuses
# inference_serving::Gptq (already-built quantization math) plus a small
# custom binary serializer, since no onnx/coremltools is installed
# locally (see npu_engine/README.md's platform notes).

add_library(NpuQuantExport quant_export.cpp)
target_include_directories(NpuQuantExport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(NpuQuantExport PUBLIC Gptq)
target_compile_features(NpuQuantExport PUBLIC cxx_std_23)
set_project_warnings(NpuQuantExport)

add_executable(npu_quant_export_test quant_export_test.cpp)
target_link_libraries(npu_quant_export_test PRIVATE NpuQuantExport Transformer)
set_project_warnings(npu_quant_export_test)

enable_testing()
add_test(NAME npu_quant_export_test COMMAND npu_quant_export_test)
