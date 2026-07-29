# Step 1: embedding model -- bidirectional encoder (reuses seq_parallel's
# LayerNorm and tensor_parallel_attn's attention primitive, unmasked)
# trained with a symmetric InfoNCE contrastive loss. Portable, header-only,
# no CUDA/Linux dependency.

add_library(RagEmbedding INTERFACE)
target_include_directories(RagEmbedding INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(RagEmbedding INTERFACE Autograd)

add_executable(embedding_model_test embedding_model_test.cpp)
target_link_libraries(embedding_model_test PRIVATE RagEmbedding Transformer)
set_project_warnings(embedding_model_test)

enable_testing()
add_test(NAME embedding_model_test COMMAND embedding_model_test)
