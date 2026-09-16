cmake_minimum_required(VERSION 3.25)

# Phase 21: HPC Storage Engineering for AI Workloads. Added 2026-09-16, not
# one of PLAN.md's original 12 phases -- see PLAN.md's "Phase 21" section
# and hpc_storage/DESIGN.md. Cross-references (does not modify)
# distributed_training/{data_loading,checkpoint,gpudirect_storage} and
# networking/{rdma_v1,nic_deep_dive,multitenancy}. Steps 3, 4, 5, 10 are
# literature/vendor-doc-grounded written analysis (README.md only, no
# CMake target); step 11 (MinIO) is real, complete, toolchain-gated code
# (no MinIO installed locally) except for its portable load-generator half.

add_subdirectory(io_pattern_characterization)
add_subdirectory(small_file_bottleneck)
add_subdirectory(storage_comparison)
add_subdirectory(checkpoint_burst_capacity)
add_subdirectory(data_reduction)
add_subdirectory(storage_qos)
add_subdirectory(dlio_bench)
add_subdirectory(minio_tuning)
