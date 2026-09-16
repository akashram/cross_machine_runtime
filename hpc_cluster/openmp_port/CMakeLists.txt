# Phase 22 step 2: OpenMP port of a hand-threaded kernel, compared against
# foundation::WorkStealingPool. Always built (the omp pragma degrades to a
# no-op/serial loop without -fopenmp -- see openmp_reduce.cpp's header
# comment) so the WorkStealingPool half of the comparison runs on this Mac
# today; linked against OpenMP::OpenMP_CXX only when find_package(OpenMP)
# (called from hpc_cluster/CMakeLists.txt) actually finds libomp.

add_executable(openmp_reduce openmp_reduce.cpp)
target_include_directories(openmp_reduce PRIVATE ${CMAKE_SOURCE_DIR}/foundation)
set_project_warnings(openmp_reduce)

if(OpenMP_CXX_FOUND)
  target_link_libraries(openmp_reduce PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME openmp_reduce_test COMMAND openmp_reduce)
