# Phase 22: HPC Cluster Systems Engineering. Added 2026-09-12, not one of
# PLAN.md's original 12 phases -- see PLAN.md's "Phase 22" section and
# hpc_cluster/DESIGN.md. Cross-references networking/ring_allreduce,
# tree_allreduce, topo_scheduler, foundation/numa, foundation/ws_pool, and
# fpga_engine/pcie_latency rather than duplicating them.
#
# Steps 3 (Slurm), 4 (health check), 7 (node design analysis), 8 (co-debug
# portfolio), 9 (Apptainer), 10 (Linux/OS tuning), and 11 (rack/facilities
# reference material) are configs/scripts/written analyses with no CMake
# target -- same convention as fpga_engine's TCL-only steps and
# cpu_engine/os_tuning's shell-script-only step.

include(CheckLanguage)

# Step 1: real MPI ring all-reduce. Hard-requires <mpi.h> to compile, so
# (like gpu_engine/'s CUDA gate) this subdirectory is only added when
# find_package(MPI) actually succeeds.
find_package(MPI)
if(MPI_CXX_FOUND)
  add_subdirectory(mpi_allreduce)
else()
  message(STATUS "MPI not found -- hpc_cluster/mpi_allreduce skipped (toolchain-gated, unrun; see that step's README)")
endif()

# Step 2: OpenMP port. Always built -- the omp pragma degrades gracefully
# to a serial no-op without -fopenmp (see openmp_port/openmp_reduce.cpp),
# so the WorkStealingPool half of the comparison runs regardless; linked
# against OpenMP::OpenMP_CXX only when found.
find_package(OpenMP)
add_subdirectory(openmp_port)

# Step 5: rack power/cooling capacity model. Pure C++, no dependency.
add_subdirectory(rack_power_model)

# Step 6: cluster reliability/MTBF-MTTR model. Pure C++, no dependency.
add_subdirectory(reliability_model)
