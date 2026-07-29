# Step 4: sentence-boundary-aware chunking + indexing pipeline. Portable,
# header-only, no CUDA/Linux dependency.

add_library(RagIndexing INTERFACE)
target_include_directories(RagIndexing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(RagIndexing INTERFACE RagEmbedding CosineANN Transformer RagCorpus)

add_executable(indexing_pipeline_test indexing_pipeline_test.cpp)
target_link_libraries(indexing_pipeline_test PRIVATE RagIndexing)
set_project_warnings(indexing_pipeline_test)

enable_testing()
add_test(NAME indexing_pipeline_test COMMAND indexing_pipeline_test)
