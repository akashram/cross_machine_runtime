# symbiyosys — formal verification of the AXI4-Stream and DMA controller RTL

**Status: formal specs complete, targeting the real RTL from step 20
(`fpga_engine/cocotb/`) — unrun, toolchain incomplete (see below).**

## What this proves
PLAN.md step 21 asks for two SymbiYosys proofs against the same RTL step
20's cocotb tests already exercise via simulation:

1. **AXI4-Stream interface never deadlocks** (`axi_nodead.sby` +
   `axi_formal.v`) — the valid/ready handshake always resolves. Three
   properties against `cocotb/axi_stream_passthrough.v`:
   - `p_hold_while_stalled` — while the output register is full and
     downstream stalls, `tvalid`/`tdata`/`tlast` must hold exactly (the
     same VALID-hold + data-stability rules `ila_debug/
     axi_trace_checker.py` checks post-hoc on one captured trace and
     cocotb's backpressure test exercises on a handful of random traces).
   - `p_no_deadlock` — once a word is accepted into the register, it
     must appear on the output exactly one cycle later (the II=1
     register-slice latency, proved instead of only observed on
     cocotb's specific 10-word throughput trace).
   - `p_ready_reflects_downstream` — whenever downstream commits to
     being ready, upstream must be ready too: no stuck backpressure,
     the "handshake always resolves" property stated directly.
2. **DMA controller never issues overlapping transactions**
   (`dma_nooverlap.sby` + `dma_formal.v`) — `mem_rden` and `mem_wren` are
   never both asserted in the same cycle, against `cocotb/
   dma_controller.v` (the same module step 20's cocotb debugging session
   found and fixed a real one-cycle-early `mem_rdata` sampling bug in).
   This holds by construction — the read states and the write state are
   mutually exclusive in the FSM — but "holds by construction" is
   exactly the kind of claim formal verification exists to machine-check
   instead of trusting by inspection.

Both `.sby` files use `mode prove` (full k-induction, not just bounded
BMC): the AXI register slice's state is a single bit plus a data
register, and the DMA controller's FSM has 7 states, so both are small
enough that k-induction should converge quickly once run, proving the
properties for every reachable state rather than a fixed search depth.

## Toolchain status
SymbiYosys needs `yosys` (RTL synthesis/formal frontend) plus the `sby`
driver script plus an SMT solver on the PATH. `z3` is already installed
on this Mac (confirmed via `which z3`), so the `.sby` files target
`smtbmc z3` rather than the boolector default from the original stub —
one less dependency to install.

`yosys` has no Homebrew bottle for this Mac's OS (macOS 13 Ventura is a
Homebrew Tier 3 / best-effort platform), so `brew install yosys` falls
back to a from-source build — and one of its dependencies, `cmake`,
_also_ has no bottle here and is itself bootstrapping from source with
LTO enabled. This is the same shape of wall the step 20 toolchain hit
with `python@3.12` (`fpga_engine/cocotb/README.md`): a slow from-source
build of a build dependency, not a fundamental blocker, deferred rather
than blocking this step indefinitely. The build was started and left
running; if it completes in a later session, `sby -f axi_nodead.sby` and
`sby -f dma_nooverlap.sby` should run directly against the real specs
above with no further changes needed.

## Results
**Toolchain** (measured locally):
```
$ which z3
/usr/local/bin/z3
$ brew install yosys
# falls back to building cmake from source (no bottle for this OS/arch) —
# still in progress, deferred rather than run to completion this session
```

**Proofs** — TODO: once `yosys` + `sby` are available:
```bash
cd fpga_engine/symbiyosys
sby -f axi_nodead.sby
sby -f dma_nooverlap.sby
```

| Proof | Result | Notes |
|---|---|---|
| `p_hold_while_stalled` | TODO | VALID-hold + data-stability while stalled |
| `p_no_deadlock` | TODO | accepted word appears on output exactly 1 cycle later |
| `p_ready_reflects_downstream` | TODO | no stuck backpressure |
| `p_no_overlapping_transactions` | TODO | `mem_rden`/`mem_wren` mutual exclusion, all reachable states |

## Files
- `axi_formal.v`, `dma_formal.v` — formal harnesses: instantiate the real
  step-20 RTL unmodified, add SVA properties as a wrapper rather than
  editing the DUT.
- `axi_nodead.sby`, `dma_nooverlap.sby` — SymbiYosys job files (`mode
  prove`, `smtbmc z3`).

## Hardware/toolchain notes
- Not FPGA-hardware-gated — like step 20, this only needs local tooling
  (yosys + sby + an SMT solver), no F1 instance.
- The RTL under test is `cocotb/axi_stream_passthrough.v` and `cocotb/
  dma_controller.v` directly (referenced via relative path from this
  directory) — not a copy — so these proofs stay in sync with whatever
  step 20's cocotb tests are simulating.
