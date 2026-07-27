cmake_minimum_required(VERSION 3.25)

# Phase 9: Inference Serving
#
# Steps 1-3 and 8 (paged_kv, continuous_batching, sla_scheduler,
# serving_backend) are pure CPU scheduling/data-structure logic with no
# GPU or Linux-specific dependency -- they build and are tested
# everywhere, including this Mac, same as this repo's other
# "portable half" steps (e.g. networking/backpressure). Steps 5-7
# (speculative_decoding, gptq, kv_quant) reuse transformer/'s real
# CPU-trained model to get real numbers instead of stubs. Step 4
# (flash_decoding) and the vLLM/TensorRT-LLM comparison half of step 9
# are genuinely GPU-gated and only build when CUDA is available (see root
# CMakeLists.txt's check_language(CUDA) gate, mirrored here).

add_subdirectory(paged_kv)
add_subdirectory(continuous_batching)
add_subdirectory(sla_scheduler)
add_subdirectory(speculative_decoding)
add_subdirectory(gptq)
add_subdirectory(kv_quant)
add_subdirectory(serving_backend)
add_subdirectory(serving_bench)

if(CMAKE_CUDA_COMPILER)
  add_subdirectory(flash_decoding)
else()
  message(STATUS "CUDA not found — inference_serving/flash_decoding skipped")
endif()
