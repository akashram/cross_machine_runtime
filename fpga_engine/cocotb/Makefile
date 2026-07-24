# Makefile — cocotb Makefile-flow runner for the two RTL modules in this
# step (axi_stream_passthrough.v, dma_controller.v).
#
# Usage:
#   make                # AXI4-Stream passthrough tests (default)
#   make TARGET=dma     # DMA controller tests
#   make SIM=verilator TARGET=dma   # override simulator (default: icarus)

SIM ?= icarus
TOPLEVEL_LANG = verilog
TARGET ?= axi

ifeq ($(TARGET),dma)
VERILOG_SOURCES = $(PWD)/dma_controller.v
TOPLEVEL = dma_controller
MODULE = test_dma
else
VERILOG_SOURCES = $(PWD)/axi_stream_passthrough.v
TOPLEVEL = axi_stream_passthrough
MODULE = test_axi
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
