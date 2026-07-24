# cocotb — Python/cocotb Makefile-driven, not CMake-compiled.
# Run: `make` (AXI4-Stream passthrough) or `make TARGET=dma` (DMA controller).
# Needs: pip install cocotb (see requirements.txt) + a Verilog simulator
# (icarus-verilog). Unlike the rest of fpga_engine/, this step doesn't need
# Vitis HLS or an F1 instance — see README.md.
