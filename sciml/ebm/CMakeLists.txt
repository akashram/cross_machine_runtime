# Phase 18 step 8: energy-based model via contrastive divergence. Depends
# on step 6's SciMLSsmLayer (finite_diff_gd_step) and step 7's
# SciMLDiffusion (for the direct same-binary architecture comparison).

add_library(SciMLEbm INTERFACE)
target_include_directories(SciMLEbm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SciMLEbm INTERFACE SciMLSsmLayer SciMLDiffusion)

add_executable(ebm_test ebm_test.cpp)
target_link_libraries(ebm_test PRIVATE SciMLEbm)
set_project_warnings(ebm_test)

enable_testing()
add_test(NAME ebm_test COMMAND ebm_test)
