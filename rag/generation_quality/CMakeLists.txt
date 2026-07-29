# Step 7: generation quality with vs. without retrieval. Portable,
# header-only, no CUDA/Linux dependency.

add_library(RagGenerationQuality INTERFACE)
target_include_directories(RagGenerationQuality INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(RagGenerationQuality INTERFACE RagIndexing ServingRouter Transformer RagCorpus)

add_executable(generation_quality_test generation_quality_test.cpp)
target_link_libraries(generation_quality_test PRIVATE RagGenerationQuality RagEmbedding)
set_project_warnings(generation_quality_test)

enable_testing()
add_test(NAME generation_quality_test COMMAND generation_quality_test)
