# Shared fixture for Phase 13 steps 4/6/7/8 -- header-only, no .cpp.
add_library(RagCorpus INTERFACE)
target_include_directories(RagCorpus INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
