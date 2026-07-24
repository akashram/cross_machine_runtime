# partial_reconfig — Dynamic Function eXchange (DFX) hot-swap

**Status: DFX hardware flow and host driver are code complete and unrun
(F1 required); the reconfiguration-time model is measured locally.**

## What this measures
Three things, deliberately kept separate since only two need an F1 instance:

1. **Reconfiguration-time model** (`reconfig_time_model.cpp`): partial
   reconfiguration time is driven by two things — the partial
   bitstream's size and the ICAP/PCAP configuration interface's
   bandwidth — this model predicts the hot-swap latency as a function of
   bitstream size, plus a fixed per-swap overhead (ICAP enable/disable,
   DRC/CRC checks, shell handshake). Compiles and runs on this Mac today
   — no Vivado needed.
2. **DFX hardware flow** (`dfx_pblock.tcl`): defines a reconfigurable
   partition (pblock) over a single kernel instance, marks it
   `HD.RECONFIGURABLE`, implements two interface-compatible kernels as
   its two configurations — `axi_stream/axi_passthrough.cpp` (RM_A,
   reused from step 4, not duplicated) and `axi_increment.cpp` (RM_B,
   new — same AXI4-Stream port interface, increments each word's data
   instead of forwarding it unchanged, so a hot-swap has an observable
   functional effect). Writes a full bitstream for configuration 1
   (static + RM_A) and a *partial* bitstream for configuration 2 (locked
   static + RM_B), then runs `pr_verify` to confirm the two
   configurations are actually safe to hot-swap between.
3. **Host hot-swap driver** (`pr_host_driver.cpp`): real XRT
   `xrt::device::load_xclbin()` calls — first the full RM_A xclbin, then
   the partial RM_B xclbin — with `std::chrono` timing around the second
   call, since that call's duration is the real "reconfiguration time"
   PLAN.md step 22 asks to measure, not the initial full-bitstream load.

## Model caveats
`kIcapBandwidthMBps` and `kFixedOverheadMs` are first-order engineering
approximations (a commonly cited UltraScale+ ICAPE3 x32 configuration
bandwidth figure and an estimated DRC/handshake overhead), not
datasheet-verified numbers for AWS F1's specific PCAP-mediated shell path
— unconfirmed without a real reconfiguration to time. What doesn't depend
on getting the constants exactly right: reconfiguration time scales
linearly with partial bitstream size plus a fixed per-swap overhead —
i.e. a smaller reconfigurable partition always reconfigures faster, which
is the actual design lever `dfx_pblock.tcl`'s pblock sizing controls.

## Results
**Reconfiguration-time model** (measured locally,
`clang++ -O2 -std=c++17 reconfig_time_model.cpp -o reconfig_time_model && ./reconfig_time_model`):

```
=== predicted partial reconfiguration time vs. bitstream size (ICAP bandwidth = 400 MB/s, fixed overhead = 5.0 ms) ===
small RM (single AXI-stream kernel, fraction of an SLR)    |  0.50 MB |    6.25 ms
medium RM (small MLP kernel, ~1 clock region)              |  2.00 MB |   10.00 ms
large RM (full SLR)                                        |  8.00 MB |   25.00 ms

applied to dfx_pblock.tcl's RM_A/RM_B pair (axi_passthrough <-> axi_increment, a single AXI4-Stream kernel, modeled at the 'small RM' size above): predicted hot-swap latency = 6.25 ms -- pr_host_driver.cpp's measured load_xclbin() duration for the partial xclbin should land near this once run on F1.
```

**Hardware** — TODO: run `dfx_pblock.tcl` to produce the real full/partial
bitstreams, package both into xclbins via `v++`, then run
`pr_host_driver.cpp <full.xclbin> <partial.xclbin>` on F1:

| Metric | Predicted | Measured | Diff |
|---|---|---|---|
| `config2_rm_b_partial.bit` size | — | TODO | — |
| Hot-swap latency (`load_xclbin` for partial xclbin) | TODO (model, at measured size) | TODO | TODO |
| `pr_verify` result | — | TODO | — |

## Files
- `reconfig_time_model.cpp` — portable, no Vivado dependency; run it directly.
- `axi_increment.cpp` — the second HLS kernel (RM_B), unrun. Deliberately
  a separate file rather than a modification of `axi_stream/
  axi_passthrough.cpp` — DFX needs two independently synthesizable RMs,
  not one kernel with a mode flag.
- `dfx_pblock.tcl` — the Vivado-side DFX flow: pblock creation, two
  configuration implementations, `pr_verify`. See file header for the
  exact command sequence and its UG909 basis.
- `pr_host_driver.cpp` — the XRT-side hot-swap + timing measurement, unrun.

## Hardware notes
- Required: AWS F1, Vitis HLS + Vivado 2022.x with a DFX-enabled license
  tier, XRT runtime.
- Run: `vivado -mode batch -source dfx_pblock.tcl -tclargs -static_checkpoint <dcp> -rm_cell <cell> -rm_a_checkpoint <dcp> -rm_b_checkpoint <dcp> -outdir <dir>`,
  then package `config1_full.bit`/`config2_rm_b_partial.bit` into xclbins
  via `v++ --package`, then `./pr_host_driver <full.xclbin> <partial.xclbin>`.
