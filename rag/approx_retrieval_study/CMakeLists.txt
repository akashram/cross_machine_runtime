# Step 8: approximate vs. exact retrieval, system-level. Pure composition
# of recall_eval (step 6) and generation_quality (step 7) -- no new
# retrieval or generation logic. Portable, no CUDA/Linux dependency.

add_executable(approx_retrieval_study_test approx_retrieval_study_test.cpp)
target_link_libraries(approx_retrieval_study_test PRIVATE RagRecallEval RagGenerationQuality RagEmbedding)
set_project_warnings(approx_retrieval_study_test)

enable_testing()
add_test(NAME approx_retrieval_study_test COMMAND approx_retrieval_study_test)
