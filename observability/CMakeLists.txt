cmake_minimum_required(VERSION 3.25)

# Phase 10: Observability, Testing, AI Integration
# eBPF requires Linux. Other steps (OTel, dashboard) are cross-platform.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "observability/ eBPF steps require Linux — building what compiles on ${CMAKE_SYSTEM_NAME}")
endif()

add_subdirectory(ebpf)
add_subdirectory(opentelemetry)
add_subdirectory(dashboard)
add_subdirectory(tlc)
add_subdirectory(symbiyosys_ci)
add_subdirectory(chaos)
add_subdirectory(nsight_agent)
add_subdirectory(kernel_variant_agent)
add_subdirectory(llm_autotune)
