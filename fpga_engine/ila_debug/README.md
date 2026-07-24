# ila_debug — ILA probe session on the AXI4-Stream interface

**Status: debug-core insertion + hardware capture is code complete and
unrun (F1 required); the protocol-violation checker is measured locally.**

## What this measures
PLAN.md step 19 asks for an ILA session on the AXI4-Stream interface that
either captures a real bug or demonstrates probing a protocol. Split the
same way as `clock_gating/` and `xadc/`, since only one half needs an F1
instance:

1. **Debug-core insertion + capture** (`ila_probes.tcl`): marks
   `axi_stream/axi_passthrough.cpp`'s `in_stream`/`out_stream`
   TVALID/TREADY/TDATA/TLAST nets for debug, inserts a real ILA IP core
   (`create_debug_core`/`connect_debug_port`/`implement_debug_core`),
   re-implements the design with it, programs the device, triggers on the
   input stream's first TVALID, and uploads a captured trace. Real Vivado
   Hardware Manager Tcl commands throughout (UG908 headless debug flow) —
   unrun, no F1 instance available locally.
2. **Protocol checker** (`axi_trace_checker.py`): the part of "reading an
   ILA capture" that doesn't need a captured trace to build — the two
   AXI4-Stream handshake rules (VALID must hold until a transfer occurs;
   the payload must stay stable while waiting) that you'd otherwise verify
   by eye in Vivado's waveform viewer, applied mechanically to a cycle
   trace instead. Its self-test runs a synthetic trace shaped like a real,
   common mistake — a hand-written AXI-Stream source with a free-running
   counter feeding TDATA, ignoring backpressure — through the checker and
   confirms it's caught. Plain Python, no Vivado dependency — passing
   output today.

This is the "demonstrate probing a protocol" branch PLAN.md's step
description explicitly allows as an alternative to "capture a real bug
during development": `axi_passthrough.cpp` is HLS-generated and simple
enough that there's no organic bug in it to catch on this Mac without
synthesis. The synthetic buggy trace stands in for that — it's the kind
of bug an ILA session on a hand-written (non-HLS) AXI-Stream block would
actually catch.

## Schema caveat
`ila_probes.tcl`'s `write_hw_ila_data -csv_file` output uses Vivado's own
CSV layout (a header block, one column per probe with a chosen radix) —
not the same as `axi_trace_checker.py`'s canonical
`cycle,tvalid,tready,tdata,tlast` schema. Reformatting one into the other
needs a real capture to look at first (documented in `ila_probes.tcl`'s
header as a TODO), so it isn't guessed at here. Also unverified: whether
`in_stream_TVALID` etc. are exactly the net names Vitis HLS emits for an
`hls::stream<ap_axiu<32,1,1,1>>` port — `ila_probes.tcl` warns rather than
failing if a probe net isn't found post-synthesis, so a real run surfaces
the correct names instead of silently probing nothing.

## Results
**Self-test** (measured locally, `python3 axi_trace_checker.py --self-test`):

```
axi_trace_checker._self_test: OK
  clean trace (axi_passthrough-shaped, backpressure for 2 cycles): 0 violations
  buggy trace (free-running-counter source, ignores backpressure): 2 violations
    cycle 2 [valid-hold]: TVALID dropped at cycle 2 while cycle 1 was still waiting for TREADY (no transfer occurred)
    cycle 2 [data-stability]: payload changed at cycle 2 (tdata 1->2, tlast 0->0) while cycle 1 was still waiting for TREADY
```

The clean trace is hand-built to the shape a correct backpressure-handling
source should produce — it is not derived from a cycle-accurate HLS
simulation of `axi_passthrough.cpp` (no synthesis tool available to
generate one), so it demonstrates the checker's logic is correct, not
that it matches `axi_passthrough`'s real post-synthesis timing.

**Hardware** — TODO: run `ila_probes.tcl` against a live F1 card with
`axi_passthrough` loaded, apply real backpressure from the testbench side
(or a second kernel), reformat the raw capture to
`axi_trace_checker.py`'s schema per the caveat above, and confirm zero
violations on the real (HLS-generated, should be correct) design:

| Capture | Cycles | Violations | Notes |
|---|---|---|---|
| in_stream, backpressure applied | TODO | TODO | TODO |
| out_stream, backpressure applied | TODO | TODO | TODO |

## Files
- `ila_probes.tcl` — hardware-gated, real Vivado Hardware Manager Tcl
  commands; unrun.
- `axi_trace_checker.py` — portable, no Vivado dependency; run it
  directly. `--self-test` needs no input files.

## Hardware notes
- Required: AWS F1, Vivado 2022.x with a connected `hw_server` (JTAG or
  the F1 shell's PCIe debug bridge — see AWS's `aws-fpga` debug docs for
  which applies to a given AFI)
- Run: `vivado -mode batch -source ila_probes.tcl -tclargs -top axi_passthrough -ip_dir <dir> -xdc <constraints> -outdir <dir>`
