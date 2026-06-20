cmake_minimum_required(VERSION 3.25)

# Phase 9: Inference Serving
# CPU development for scheduler/paging logic; GPU required for latency benchmarks.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "inference_serving/ requires Linux — skipping on ${CMAKE_SYSTEM_NAME}")
  return()
endif()

add_subdirectory(paged_kv)
add_subdirectory(continuous_batching)
add_subdirectory(sla_scheduler)
add_subdirectory(flash_decoding)
add_subdirectory(speculative_decoding)
add_subdirectory(gptq)
add_subdirectory(kv_quant)
add_subdirectory(serving_backend)
add_subdirectory(serving_bench)
