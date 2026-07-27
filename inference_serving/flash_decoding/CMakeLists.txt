# inference_serving/flash_decoding — Step 4: split-K decode-time attention.
# Only added by the parent CMakeLists.txt when CMAKE_CUDA_COMPILER is
# found (mirrors root CMakeLists.txt's gpu_engine gate).
#
# flash_decoding: INTERFACE library exposing flash_decoding.cuh
#   (header-only device code; same online-softmax design as
#   gpu_engine/flash_attn's kernel, no code dependency on it).
#
# flash_decoding_bench: compares num_kv_splits=1 (decode-time degenerate
#   case of flash_attn's query-tile grid) against the auto-chosen split
#   count, latency across N in {512..32768} -- PLAN.md's "8k+ token
#   sequences" ask.

add_library(flash_decoding_lib INTERFACE)
target_include_directories(flash_decoding_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(flash_decoding_bench flash_decoding_bench.cu)
target_apply_cuda_flags(flash_decoding_bench)
target_link_libraries(flash_decoding_bench PRIVATE CUDA::cudart flash_decoding_lib)
set_target_properties(flash_decoding_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
