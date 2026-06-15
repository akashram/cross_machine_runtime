add_executable(device_query device_query.cu)
target_apply_cuda_flags(device_query)
target_link_libraries(device_query PRIVATE CUDA::cudart)
set_target_properties(device_query PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
