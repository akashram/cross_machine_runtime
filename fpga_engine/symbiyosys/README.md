# symbiyosys — formal verification of the AXI4-Stream and DMA controller RTL

**Status: complete — both proofs run locally against the real RTL from
step 20 (`fpga_engine/cocotb/`) and PASS by k-induction (see Results).**

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
back to a from-source build — and one of its dependencies, `cmake` (and
then `tcl-tk`), _also_ has no bottle here and bootstraps from source with
LTO enabled: the same shape of wall the step 20 toolchain hit with
`python@3.12`. That build was left running for ~2 hours across a session
without reaching `yosys` itself. Rather than keep paying that cost,
switched to YosysHQ's prebuilt **OSS CAD Suite** release
(`oss-cad-suite-darwin-x64-<date>.tgz` from
`github.com/YosysHQ/oss-cad-suite-build`) — a ~460 MiB tarball with
`yosys`/`sby`/solvers prebuilt for macOS x64, extracted to
`~/oss-cad-suite` and put on `PATH` via `source ~/oss-cad-suite/environment`.
Working `yosys`/`sby` in minutes instead of hours; the abandoned brew
build was killed cleanly (whole build process group, `kill -TERM` on the
group PID) with no Cellar corruption, since Homebrew only moves a build
into the Cellar on success.

**A real syntax gap found along the way:** the free OSS CAD Suite's
`yosys` does not parse full SVA `assert property (@(posedge clk)
disable iff (...) ... |=> ...)` syntax — that clocking-event/
disable-iff/temporal-implication grammar needs YosysHQ's commercial
Verific frontend plugin, confirmed by the suite's own bundled
`examples/fifo/fifo.sv`, which has an `` `ifdef VERIFIC `` branch using
exactly that syntax and an `` `else `` branch with plain procedural
asserts for when Verific isn't available. `axi_formal.v` and
`dma_formal.v` were rewritten from SVA property syntax into yosys-native
procedural `always @(posedge clk) if (...) assert(...)` form using
`$past`/`$stable` as plain functions (confirmed working via a minimal
standalone test) — same properties, same k-induction proof strength
under `mode prove`, different (free-toolchain-compatible) syntax.

**Two real formal-verification bugs found and fixed, not RTL bugs:**
1. `axi_formal.v`'s first pass (before adding an `$initstate` guard)
   reported `p_no_deadlock` failing at step 1. Tracing the counterexample
   VCD showed the failure was driven entirely by an unconstrained
   `$past()` value at the fictitious initial state (no real "cycle -1"
   exists at the first sampled step, so `$past()` there is a free
   variable the solver can pick to spuriously satisfy an antecedent).
   Fixed by gating the `$past()`-dependent checks with `if (!$initstate
   && ...)`, yosys's native "true only at the fictitious initial state"
   function — the standard idiom for this exact pitfall.
2. `dma_formal.v`'s first pass reported `p_no_overlapping_transactions`
   failing at step 1 with a *different* root cause: no `$past()` is
   involved here, but `rst_n` is a free top-level input the solver can
   hold at 1 from the very first cycle, so `state`/`mem_rden`/`mem_wren`
   (plain regs with no explicit initial value) start from an arbitrary
   "anyinit" value instead of a real post-reset one. The traced
   counterexample was `state=101` (S_NEXT) with `mem_rden=1` **and**
   `mem_wren=1` simultaneously at cycle 0 — a combination no reachable
   FSM transition produces, only reachable as an ungrounded power-on
   garbage state. Fixed with `initial assume(!rst_n);`, forcing the
   basecase to start from an actual reset rather than an arbitrary one.

Both are the same underlying formal-verification lesson from two
different angles: an unconstrained "before time began" value (past
history in case 1, initial register contents in case 2) will produce
counterexamples that are artifacts of the proof setup, not the design —
worth documenting since it's exactly the kind of mistake "holds by
construction" claims (this step's whole premise) can hide if the formal
harness itself isn't checked as carefully as the RTL.

## Results
**Toolchain** (measured locally):
```
$ source ~/oss-cad-suite/environment
$ yosys --version
Yosys 0.67+92 (git sha1 30fe16c7f-dirty, Release, ...)
$ sby --version
SBY v0.67-4-gfea6e46
```

**Proofs** (measured locally, clean rerun from scratch):
```bash
cd fpga_engine/symbiyosys
source ~/oss-cad-suite/environment
sby -f axi_nodead.sby      # DONE (PASS, rc=0) -- successful proof by k-induction
sby -f dma_nooverlap.sby   # DONE (PASS, rc=0) -- successful proof by k-induction
```

| Proof | Result | Notes |
|---|---|---|
| `p_hold_while_stalled` | **PASS** | VALID-hold + data-stability while stalled |
| `p_no_deadlock` | **PASS** | accepted word appears on output exactly 1 cycle later ($initstate guard needed, see above) |
| `p_ready_reflects_downstream` | **PASS** | no stuck backpressure |
| `p_no_overlapping_transactions` | **PASS** | `mem_rden`/`mem_wren` mutual exclusion, all reachable states (reset assumption needed, see above) |

Both `.sby` files use `mode prove`: every reported engine status above is
`pass` for **both** basecase and induction, i.e. a genuine full
k-induction proof (holds for every reachable state, not a bounded
search), not just a BMC pass at a fixed depth.

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
