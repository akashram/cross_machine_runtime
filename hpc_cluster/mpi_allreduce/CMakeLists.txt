# Phase 22 step 1: real MPI ring all-reduce. Only added to the build when
# find_package(MPI) succeeds (hpc_cluster/CMakeLists.txt gates this
# subdirectory the same way root CMakeLists.txt gates gpu_engine/ behind
# check_language(CUDA)).

add_executable(mpi_ring_allreduce mpi_ring_allreduce.cpp)
target_link_libraries(mpi_ring_allreduce PRIVATE MPI::MPI_CXX)
set_project_warnings(mpi_ring_allreduce)

if(MPIEXEC_EXECUTABLE)
  add_test(NAME mpi_ring_allreduce_test
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                   $<TARGET_FILE:mpi_ring_allreduce>)
endif()
