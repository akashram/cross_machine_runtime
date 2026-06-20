cmake_minimum_required(VERSION 3.25)

# Phase 7: FPGA Backend
# Vitis HLS projects are built via TCL scripts, not CMake.
# CMake here only manages documentation targets and the test harnesses.
# To synthesize: run tcl_pipeline/synth.tcl in Vivado headless mode.

if(NOT DEFINED XILINX_VITIS)
  message(STATUS "XILINX_VITIS not set — fpga_engine/ stub only (no synthesis)")
  # Still add subdirectories so cocotb/symbiyosys stubs are visible
endif()

add_subdirectory(f1_setup)
add_subdirectory(tcl_pipeline)
add_subdirectory(power_ci)
add_subdirectory(axi_stream)
add_subdirectory(dot_product)
add_subdirectory(loop_opt)
add_subdirectory(dsp_lut)
add_subdirectory(fixed_point)
add_subdirectory(bram_uram)
add_subdirectory(ddr4)
add_subdirectory(dma)
add_subdirectory(pcie_latency)
add_subdirectory(pingpong)
add_subdirectory(ml_kernel)
add_subdirectory(timing_closure)
add_subdirectory(slr)
add_subdirectory(clock_gating)
add_subdirectory(xadc)
add_subdirectory(ila_debug)
add_subdirectory(cocotb)
add_subdirectory(symbiyosys)
add_subdirectory(partial_reconfig)
add_subdirectory(fpga_net)
add_subdirectory(vitis_ai)
add_subdirectory(thermal_router)
