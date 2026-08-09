# Phase 18 step 7: DDPM diffusion on a toy 2D distribution. Depends on
# step 6's SciMLSsmLayer (reuses finite_diff_gd_step).

add_library(SciMLDiffusion INTERFACE)
target_include_directories(SciMLDiffusion INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SciMLDiffusion INTERFACE SciMLSsmLayer)

add_executable(diffusion_test diffusion_test.cpp)
target_link_libraries(diffusion_test PRIVATE SciMLDiffusion)
set_project_warnings(diffusion_test)

enable_testing()
add_test(NAME diffusion_test COMMAND diffusion_test)
