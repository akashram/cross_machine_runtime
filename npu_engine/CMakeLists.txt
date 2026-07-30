# Phase 15: NPU Backend.
# Portable pieces (quant_export) are wired here. cost_model/, op_coverage/,
# and thermal/ follow the fpga_engine/tpu_engine "*_model.cpp, run
# manually via clang++, output captured in README" convention instead of
# CMake/ctest -- see each subdirectory's README for the exact build
# command and captured output. Step 1 (toolchain validation) is
# hardware-gated and has no local artifact at all (see README.md).

add_subdirectory(quant_export)
