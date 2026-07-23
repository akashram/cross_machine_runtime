# axi_stream

**Status: code complete — requires Vitis HLS on F1 to synthesize/cosimulate.**

## What this validates
`axi_passthrough.cpp` is the smallest possible AXI4-Stream kernel: read a
word, write it back unchanged, stop on `TLAST`. Its only purpose is to
validate the AXI4-Stream protocol path itself — `s_axilite` control
handshake plus `axis` data handshake — before any real compute kernel
(dot_product, ml_kernel, ...) builds on top of it. `ap_axiu<32,1,1,1>`
carries a 32-bit `TDATA` word with 1-bit `TUSER`/`TID`/`TDEST`, matching
the narrowest AXI4-Stream word Vitis HLS will generate control logic for.

The `do { ... } while (!word.last)` shape (rather than a `for`) matters:
AXI4-Stream packets have no length prefix, only an in-band `TLAST` flag, so
the kernel must consume at least one word before it can know whether to
stop — a `for`-loop bound on a known `n` would be wrong for stream
protocols in general, even though it's exactly right for dot_product's
DDR4-backed `m_axi` ports where the length **is** known upfront.

`#pragma HLS PIPELINE II=1` targets one word in/out per cycle — the
baseline every downstream streaming kernel is compared against.

## Results
TODO: cosimulate + synthesize on F1.

| Metric | Value |
|--------|-------|
| II | TODO |
| Fmax (MHz) | TODO |
| LUT / FF | TODO |
| Cosim: data integrity (output == input) | TODO |
| Cosim: TLAST correctly terminates | TODO |

## Hardware notes
- Required: Vitis HLS 2022.x (`vitis_hls -f run_hls.tcl`, cosim with `-rtl verilog`)
- See `fpga_engine/cocotb/` and `fpga_engine/symbiyosys/` for RTL-level
  testbench and formal verification of this same protocol, independent of
  Vitis HLS cosimulation.
