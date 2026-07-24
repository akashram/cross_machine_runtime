# cocotb — Python RTL testbenches for AXI4-Stream and DMA controller

**Status: complete and run locally. 4/4 tests pass (Icarus Verilog +
cocotb 2.0.1, Python 3.12). A real off-by-one bug in the DMA controller's
read timing was caught and fixed along the way — see below.**

## What this measures
PLAN.md step 20 asks for cocotb (Python-based RTL simulation) tests
against the AXI stream interface and the DMA controller. Unlike most of
`fpga_engine/`, this step's dependency isn't Vivado/Vitis HLS/an F1
instance — cocotb runs against an open-source Verilog simulator
(Icarus Verilog), so in principle this is portable and runnable on this
Mac without any AWS hardware at all. Two RTL modules + testbenches:

1. **`axi_stream_passthrough.v`** — a hand-written RTL model of
   `axi_stream/axi_passthrough.cpp`'s behavior (that file is HLS C++;
   turning it into simulatable RTL needs Vitis HLS, unavailable here — see
   that step's README). Standard single-register AXI4-Stream slice.
   `test_axi.py` drives it with (a) back-to-back words, no backpressure,
   checking data integrity + TLAST alignment + II=1 throughput, and (b)
   randomized stalls on both source and sink sides wrapped in
   `with_timeout`, so an actual protocol deadlock fails the test instead
   of hanging forever — the "handshake always resolves" property
   `ila_debug/`'s step describes checking by eye on a real capture,
   checked here by running the real RTL against a random stall pattern.
2. **`dma_controller.v`** — a hand-written single-outstanding DMA engine
   (read word, buffer it, write word, repeat), also the target of step
   21's SymbiYosys "never issues overlapping transactions" proof.
   `test_dma.py` drives it against a Python-modeled synchronous memory,
   checks the copy is byte-exact, and (in
   `test_dma_no_overlapping_transactions`) asserts inside the memory
   model itself if `mem_rden`/`mem_wren` are ever both asserted in the
   same cycle — the same property step 21 proves statically for every
   reachable state, spot-checked here at runtime for the traces this test
   happens to drive.

## Toolchain status
Icarus Verilog (`iverilog`, v13.0) is installed via Homebrew and
confirmed working. This Mac's only Homebrew Python was initially 3.14,
and cocotb 2.0.1 declares a hard maximum of Python 3.13
(`RuntimeError: cocotb 2.0.1 only supports a maximum Python version of
3.13` from its build backend). `brew install python@3.12` (a slow
PGO+LTO CPython build plus an `openssl@3` source build) finished in a
later session; with `python3.12` available, `pip install cocotb` and both
Makefile flows ran cleanly.

## A real bug caught by running this
The DMA controller's first version had only one wait state between
asserting `mem_rden` and sampling `mem_rdata` — one cycle too few for
any same-clock-domain requester/synchronous-memory pair: `mem_rden`/
`mem_addr` become stable starting at the edge that asserts them (E0), a
registered memory samples them and produces `mem_rdata` starting at the
*following* edge (E1, stable through [E1, E2)), so the requester can
only safely sample at E2 — not E1. The single-wait-state version sampled
at E1, one cycle early. Running `test_dma_copy` against cocotb's
posedge-synchronous memory model (not just reasoning about the FSM by
inspection) exposed it immediately as a one-word lag in every copied
value:
```
AssertionError: copy mismatch: src=[40960, 40961, 40962, 40963, 40964, 40965, 40966, 40967]
dst=[0, 40960, 40961, 40962, 40963, 40964, 40965, 40966]
```
Fixed by splitting the wait into two cycles (`S_READ_WAIT1`,
`S_READ_WAIT2` in `dma_controller.v`) so there's a full extra cycle of
margin before sampling `mem_rdata`. Re-run confirmed the fix — see
results below.

## Results
**Toolchain** (measured locally):
```
$ iverilog -V
Icarus Verilog version 13.0 (stable) (v13_0)
$ python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
Successfully installed cocotb-2.0.1 ...
```

**Tests** (measured locally, Icarus Verilog + cocotb 2.0.1):
```bash
cd fpga_engine/cocotb
source .venv/bin/activate
make                # axi_stream_passthrough tests
make TARGET=dma      # dma_controller tests
```
```
** TEST                                           STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
** test_axi.test_axi_passthrough_full_throughput   PASS          30.00           0.00      11472.39  **
** test_axi.test_axi_passthrough_backpressure      PASS         156.00           0.01      13869.29  **
** TESTS=2 PASS=2 FAIL=0 SKIP=0                                 186.00           0.02      12103.15  **

** TEST                                           STATUS  SIM TIME (ns)  REAL TIME (s)  RATIO (ns/s) **
** test_dma.test_dma_copy                          PASS          94.00           0.01      17286.24  **
** test_dma.test_dma_no_overlapping_transactions   PASS         176.00           0.01      20017.83  **
** TESTS=2 PASS=2 FAIL=0 SKIP=0                                 270.00           0.02      17063.95  **
```

| Test | Result | Notes |
|---|---|---|
| `test_axi_passthrough_full_throughput` | PASS | data integrity + II=1 timing (10 words in 30ns @ 2ns period) |
| `test_axi_passthrough_backpressure` | PASS | randomized stalls on both sides, deadlock timeout guard |
| `test_dma_copy` | PASS | byte-exact copy check (after the read-timing fix above) |
| `test_dma_no_overlapping_transactions` | PASS | runtime spot-check of step 21's formal property |

## Files
- `axi_stream_passthrough.v`, `dma_controller.v` — hand-written RTL, no
  Vitis HLS dependency.
- `test_axi.py`, `test_dma.py` — cocotb testbenches.
- `Makefile` — cocotb Makefile-flow runner (`make` / `make TARGET=dma`).
- `requirements.txt` — `cocotb` (needs Python <= 3.13).

## Hardware notes
- Not hardware-gated in the usual `fpga_engine/` sense — no F1 instance
  needed, just a cocotb-compatible Python (<=3.13) plus a Verilog
  simulator (Icarus Verilog, installed) or Verilator.
- On F1/Linux, the same testbenches could instead target the real
  HLS-generated RTL (post `vitis_hls` export) via Vivado's XSIM simulator
  (`SIM=xcelium`/`SIM=xsim` if cocotb's Vivado backend is used) instead of
  the hand-written stand-ins here — noted as a follow-up, not required by
  this step.
