# Phase 21 step 11: MinIO hands-on tuning. minio_load_gen is the portable
# half -- real, builds and runs today. deploy_minio.sh/tune_and_measure.sh
# are real, complete, but toolchain-gated (no MinIO installed locally).

add_executable(minio_load_gen minio_load_gen.cpp)
target_link_libraries(minio_load_gen PRIVATE DataLoading)
target_include_directories(minio_load_gen PRIVATE ${CMAKE_SOURCE_DIR}/distributed_training/checkpoint)
set_project_warnings(minio_load_gen)

enable_testing()
add_test(NAME minio_load_gen COMMAND minio_load_gen /tmp/hpc_storage_minio_load_ctest)
