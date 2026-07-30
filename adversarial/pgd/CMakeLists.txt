# Step 3: PGD. Portable, header-only, no CUDA/Linux dependency.

add_library(AdversarialPGD INTERFACE)
target_include_directories(AdversarialPGD INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(AdversarialPGD INTERFACE AdversarialFGSM)

add_executable(pgd_test pgd_test.cpp)
target_link_libraries(pgd_test PRIVATE AdversarialPGD)
set_project_warnings(pgd_test)

enable_testing()
add_test(NAME pgd_test COMMAND pgd_test)
