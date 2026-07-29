# Phase 13: Retrieval-Augmented Generation. Added 2026-07-28, not one of
# PLAN.md's original 12 phases -- see PLAN.md's "Phase 13" section and
# rag/DESIGN.md. Fully CPU-portable: builds on transformer/ (generation),
# ml/knn (ANN retrieval), and inference_serving (serving path), all
# already code-complete. No CUDA/Linux/MLIR dependency for any step.

add_subdirectory(corpus)
add_subdirectory(embedding_model)
add_subdirectory(hnsw)
add_subdirectory(indexing_pipeline)
add_subdirectory(rag_generation)
add_subdirectory(recall_eval)
