# Step 5: retrieval-augmented prompt construction + generation wiring.
# Portable, header-only, no CUDA/Linux dependency.

add_library(RagGeneration INTERFACE)
target_include_directories(RagGeneration INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(RagGeneration INTERFACE RagIndexing ServingRouter Transformer)

add_executable(rag_generation_test rag_generation_test.cpp)
target_link_libraries(rag_generation_test PRIVATE RagGeneration RagCorpus)
set_project_warnings(rag_generation_test)

enable_testing()
add_test(NAME rag_generation_test COMMAND rag_generation_test)
