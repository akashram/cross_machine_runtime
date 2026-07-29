# Step 6: recall@k against a real relevance-labeled query set. Portable,
# header-only, no CUDA/Linux dependency.

add_library(RagRecallEval INTERFACE)
target_include_directories(RagRecallEval INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(RagRecallEval INTERFACE RagIndexing RagCorpus)

add_executable(recall_eval_test recall_eval_test.cpp)
target_link_libraries(recall_eval_test PRIVATE RagRecallEval RagEmbedding)
set_project_warnings(recall_eval_test)

enable_testing()
add_test(NAME recall_eval_test COMMAND recall_eval_test)
