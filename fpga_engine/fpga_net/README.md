# fpga_net — RDMA-like direct network access, bypassing the host CPU

**Status: P4 pipeline and OpenNIC shell integration are code complete and
unrun (F1 + P4 toolchain required); the latency comparison model is
measured locally.**

## What this measures
Three things, deliberately kept separate since two need an F1 instance
and a P4 toolchain neither of which exist locally:

1. **Latency comparison model** (`net_latency_model.cpp`): a first-order
   latency budget for a small one-sided WRITE over each path. Both paths
   share the same physical network hop; what differs is everything after
   the packet arrives at the destination NIC — the FPGA-bypass path
   (`rdma_bypass.p4`) parses, matches, and DMAs the payload entirely
   inside the P4 pipeline at line rate, while the CPU-mediated path
   (a standard kernel socket) pays for NIC-to-host DMA, interrupt/NAPI
   dispatch, kernel network-stack traversal, a user-space copy, and a
   context switch. Compiles and runs on this Mac today — no F1 or P4
   toolchain needed.
2. **P4 pipeline** (`rdma_bypass.p4`): a minimal RDMA-like packet
   pipeline targeting a P4-programmable NIC (OpenNIC shell's user-plugin
   box, compiled via `p4c-xsa`). Parses Ethernet/IPv4/UDP plus a
   lightweight RDMA-style header (opcode, remote address, length) and, on
   a WRITE opcode, dispatches straight to a DMA-engine action in the same
   pipeline pass — no table or path here hands the packet to host
   software before the payload lands, which is the actual mechanism
   "bypassing the host CPU" refers to.
3. **OpenNIC shell integration** (`onic_shell_integration.tcl`): wires
   `rdma_bypass.p4`'s compiled pipeline into the shell's user-plugin box
   — AXI4-Stream RX from the CMAC, AXI4-Stream TX to the DMA engine,
   AXI4-Lite control for read-only counters — via the shell's IP
   Integrator block design.

## Model caveats
Every per-stage constant in `net_latency_model.cpp` (P4 pipeline latency,
interrupt dispatch, kernel stack traversal, etc.) is a commonly cited
order-of-magnitude figure for a modern x86 Linux kernel network stack and
a P4-programmable NIC pipeline — the kind of numbers kernel-bypass
literature (DPDK, RDMA vendor whitepapers) consistently reports, not
datasheet-verified numbers for this specific F1 shell/instance pairing,
unconfirmed without a real measurement. What doesn't depend on getting
the constants exactly right: the FPGA-bypass path structurally skips
every CPU-mediated-path stage by construction, so its latency floor is
lower regardless of the exact numbers, and the gap should be dominated by
kernel-stack-traversal + interrupt-dispatch + context-switch — not the
shared network hop — which is a falsifiable, checkable claim once real
numbers exist.

## Results
**Latency comparison model** (measured locally,
`clang++ -O2 -std=c++17 net_latency_model.cpp -o net_latency_model && ./net_latency_model`):

```
=== predicted one-sided WRITE latency: FPGA-bypass vs. CPU-mediated ===
FPGA-bypass (rdma_bypass.p4):   network=0.50us + P4 pipeline=0.15us + on-chip DMA handoff=0.10us = 0.75us
CPU-mediated (kernel socket):   network=0.50us + NIC->host DMA=0.80us + interrupt dispatch=2.00us + kernel stack=3.50us + user copy=1.50us + context switch=2.00us = 10.30us

predicted speedup = 13.73x (9.55us saved, 92.7% of the CPU-mediated total)
the gap is dominated by kernel-stack-traversal + interrupt-dispatch + context-switch (7.50us of 10.30us, 72.8%) -- the exact stages the FPGA-bypass path has no equivalent of, not the shared network hop.
```

**Hardware** — TODO: compile `rdma_bypass.p4` with `p4c-xsa`, integrate
via `onic_shell_integration.tcl`, build the OpenNIC shell bitstream, and
measure real WRITE latency against an equivalent kernel-socket baseline
on two F1 instances:

| Path | Predicted | Measured | Diff |
|---|---|---|---|
| FPGA-bypass (rdma_bypass.p4) | 0.75 us | TODO | TODO |
| CPU-mediated (kernel socket) | 10.30 us | TODO | TODO |
| Speedup | 13.73x | TODO | TODO |

## Files
- `net_latency_model.cpp` — portable, no F1/P4 dependency; run it directly.
- `rdma_bypass.p4` — the P4_16 packet pipeline (parser + bypass-dispatch
  table + deparser), unrun. See file header for the OpenNIC-shell target
  and toolchain caveat.
- `onic_shell_integration.tcl` — wires the compiled pipeline into the
  OpenNIC shell block design, unrun. See file header for the exact
  command sequence and its toolchain caveat.

## Hardware notes
- Required: AWS F1, OpenNIC shell (or Vitis Networking P4) AFI, `p4c-xsa`
  P4 compiler, two F1 instances in the same placement group to measure a
  real network hop between them (same topology `networking/`'s Phase 5
  steps use for multi-node measurements).
- Run: `p4c-xsa rdma_bypass.p4 -o <ip_dir>`, then
  `vivado -mode batch -source onic_shell_integration.tcl -tclargs -bd_design <name> -p4_ip_repo <ip_dir> -p4_ip_vlnv <vlnv> -box_inst rdma_bypass_box -outdir <dir>`.
