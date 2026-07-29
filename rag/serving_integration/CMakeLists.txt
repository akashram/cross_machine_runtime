# Step 9: wires retrieval + generation into inference_serving's
# ServingRouter as a new request path, without modifying serving_router.h/
# .cpp itself. Portable, header-only, no CUDA/Linux dependency.

add_library(RagServingIntegration INTERFACE)
target_include_directories(RagServingIntegration INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(RagServingIntegration INTERFACE RagIndexing RagGeneration ServingRouter Transformer)

add_executable(serving_integration_test serving_integration_test.cpp)
target_link_libraries(serving_integration_test PRIVATE RagServingIntegration RagCorpus)
set_project_warnings(serving_integration_test)

enable_testing()
add_test(NAME serving_integration_test COMMAND serving_integration_test)
