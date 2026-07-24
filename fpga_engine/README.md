# Phase 7: FPGA Backend

**Status: CODE COMPLETE (25/25 steps). No AWS F1 instance on Mac, so
every step is code-complete and locally runnable wherever it doesn't
strictly need Vivado/Vitis HLS/an FPGA card, with the hardware-only piece
written as real (not stub) TCL/HLS/XRT code, unrun — see each step's own
README for which half is measured vs. TODO, and `CLAUDE.md`'s Phase 7
section for the full per-step summary.**

## Overview
HLS kernels, AXI interfaces, DMA orchestration, timing closure, partial
reconfiguration, FPGA network stack, and formal verification via SymbiYosys.

## Build model
FPGA synthesis is TCL-driven (Vivado headless), not CMake. CMake here manages
cocotb testbenches and SymbiYosys formal proofs only.

To synthesize on F1 instance:
```bash
vivado -mode batch -source fpga_engine/tcl_pipeline/synth.tcl
```

## Steps

| # | Directory | What |
|---|-----------|------|
| 1 | f1_setup | F1 instance setup, xbutil validate |
| 2 | tcl_pipeline | Fully scriptable synth + impl + bitstream |
| 3 | power_ci | Vivado power report per bitstream in CI |
| 4 | axi_stream | AXI4-Stream passthrough kernel |
| 5 | dot_product | First HLS kernel, II=1 pipelining |
| 6 | loop_opt | UNROLL / PIPELINE / DATAFLOW comparison |
| 7 | dsp_lut | DSP48E2 vs LUT tradeoff analysis |
| 8 | fixed_point | ap_fixed<W,I> precision/resource tradeoffs |
| 9 | bram_uram | BRAM vs URAM access pattern analysis |
| 10 | ddr4 | DDR4 integration, multi-bank bandwidth |
| 11 | dma | DMA orchestration, interrupt vs poll |
| 12 | pcie_latency | PCIe latency component decomposition |
| 13 | pingpong | Double-buffer compute/transfer overlap |
| 14 | ml_kernel | MLP inference: II=1, INT8, pipelined |
| 15 | timing_closure | Critical path analysis, Fmax targets |
| 16 | slr | SLR partitioning, crossing penalty |
| 17 | clock_gating | Clock enables, dynamic power reduction |
| 18 | xadc | Temperature + voltage rail monitoring |
| 19 | ila_debug | ILA probe session, protocol debugging |
| 20 | cocotb | Python RTL testbenches (AXI stream, DMA) |
| 21 | symbiyosys | Formal verification: AXI no-deadlock |
| 22 | partial_reconfig | Hot-swap kernels at runtime |
| 23 | fpga_net | OpenNIC/Vitis Networking P4 RDMA stack |
| 24 | vitis_ai | Vitis AI vs custom HLS comparison |
| 25 | thermal_router | Die temp → automatic load reduction |

## Hardware notes
- Required: AWS F1 instance (f1.2xlarge or f1.4xlarge), Xilinx Vitis AMI
- Spot pricing: ~$0.50–$1/hr
