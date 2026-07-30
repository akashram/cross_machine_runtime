# Shared classification-task fixture for Phase 14 -- header-only, no .cpp.
add_library(AdversarialTask INTERFACE)
target_include_directories(AdversarialTask INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AdversarialTask INTERFACE Autograd)
