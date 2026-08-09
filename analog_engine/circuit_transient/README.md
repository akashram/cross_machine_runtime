# circuit_transient

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 17 step 8 (the phase's last step): an analog circuit
transient surrogate — settling time and bandwidth vs. parasitic R/C for a
crossbar bitline, same disclosed-simplification pattern as
`fpga_engine/thermal_router`'s first-order RC step-response model
(temperature there, voltage here — identical math).

## Design

- **The real gap this closes**: step 2's crossbar MAC simulation treats a
  cell read as instantaneous — no settling time at all. This step adds
  that back as a real, separate physical cost, deliberately scoped out of
  step 2 (this repo's convention: one real, disclosed simplification added
  at a time, not one step trying to model everything).
- **R/C scale with crossbar size via a distributed (Elmore-style) line
  model**: each of `N` cells along a bitline contributes its own wire-
  segment resistance and parasitic capacitance to the shared line, so
  `tau(N) = R_per_cell * C_per_cell * N^2` — tau grows QUADRATICALLY with
  crossbar size. This is the standard real result for a distributed RC
  line (Elmore delay), not an invention for this repo.
- `R_per_cell`/`C_per_cell` are illustrative literature-order-of-magnitude
  constants (advanced-node on-chip wire resistance/parasitic capacitance
  per cell-pitch), not measurements — no fab access, same disclosed
  limitation as every other Phase 17 step.

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  V(t=tau)/V_final = 0.632121 (theory: 1 - 1/e = 0.632121)
PASS  step_response at t=tau matches the textbook 1-1/e fraction exactly
  settling_time_seconds(0.99 threshold) = 2.3578e-10 s -> step_response there = 0.990000 (target: 0.99)
PASS  the closed-form settling time, fed back into step_response, lands exactly at the target threshold fraction
  tau(32)=5.1200e-11 s, tau(64)=2.0480e-10 s, ratio=4.0000 (theory: exactly 4.0, quadratic in crossbar size)
PASS  tau scales EXACTLY quadratically with crossbar size (doubling size -> 4x tau), the real distributed-RC-line result

  crossbar bitline RC step response by size (r_per_cell=50 ohm, c_per_cell=1.0e-15 F):
      size            tau settling_time(99%)        bandwidth
         8   3.200e-12 s       1.474e-11 s     4.974e+10 Hz
        16   1.280e-11 s       5.895e-11 s     1.243e+10 Hz
        32   5.120e-11 s       2.358e-10 s     3.108e+09 Hz
        64   2.048e-10 s       9.431e-10 s     7.771e+08 Hz
       128   8.192e-10 s       3.773e-09 s     1.943e+08 Hz
```

## Findings

- **A third, independent reason "bigger crossbar" isn't free — layered on
  top of step 2's own finding.** Step 2 found crossbar size doesn't
  improve MAC accuracy for random weights (signal and noise both scale
  as `sqrt(M)`). This step adds a genuinely different, orthogonal cost:
  even holding accuracy fixed, a bigger crossbar has a QUADRATICALLY
  longer settling time (`3.2ps` at size 8 vs. `0.82ns` at size 128 — a
  256x increase for a 16x size increase, exactly `16^2`) and
  correspondingly lower bandwidth (`49.7 GHz` equivalent down to
  `194 MHz`, a ~256x drop). Two independently-motivated models in this
  phase (step 2's accuracy-vs-size sweep, this step's physics-vs-size
  model) both conclude bigger crossbars carry real costs without a
  matching accuracy benefit — from completely different mechanisms
  (statistics vs. circuit physics), which makes the combined conclusion
  more load-bearing than either alone.
- All three closed-form circuit-theory relationships checked out exactly
  in code, not just in the comment: the textbook `1 - 1/e` fraction at
  `t=tau` matched to `1e-9` precision, the settling-time formula's
  self-consistency check landed exactly on its target threshold, and the
  quadratic size-scaling ratio measured exactly `4.0000` for a size
  doubling — all real, deterministic checks of the implemented formulas,
  not assumed correct because they look right on paper.
- At the largest size this repo's crossbar work has tested (128, matching
  step 6's PE-array sweep), settling time is `3.77ns` — small in absolute
  terms, but real: it sets a genuine per-read latency floor a physical
  crossbar accelerator has to budget for, on top of step 4's per-MAC
  energy cost, that step 2's original (instantaneous-read) model didn't
  account for at all.

## Hardware notes
None — pure CPU. No fab access to measure real bitline RC directly (see
PLAN.md's Phase 17 hardware-access note); `R_per_cell`/`C_per_cell` are
illustrative order-of-magnitude constants.
