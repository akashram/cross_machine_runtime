# Phase 17: Analog & Unconventional Compute Hardware

**Status: CODE COMPLETE (8/8 steps), 2026-08-09.** Unlike Phase 3/7/8/15's
hardware-gated-but-locally-codeable split, this phase has NO toolchain to
gate behind at all — no analog/neuromorphic silicon exists to rent from
any cloud provider — so every step is real, locally-runnable numeric/
simulation code, actually compiled, run, and captured in each step's own
README.

## Overview

Scoped 2026-08-09 while comparing this repo against a batch of analog/
physics-based-AI-compute job descriptions, which found a clean zero on
analog computing, non-volatile memory devices, and PPA/dataflow
accelerator-design tooling — despite this repo's deep coverage of digital
accelerator architecture (GPU/FPGA/TPU/NPU). This phase closes that gap.

## Steps

| # | Directory | What |
|---|-----------|------|
| 1 | `device_model` | RRAM-like conductance-cell non-ideality model: write/read noise, power-law drift, endurance-linked failure |
| 2 | `crossbar_mac` | Resistive crossbar analog matrix-vector multiply (Ohm's law + KCL), signed weights via differential G+/G- pairs |
| 3 | `nvm_comparison` | RRAM vs. PCM vs. STT-MRAM vs. SRAM-CIM tradeoff comparison, literature-grounded figure-of-merit |
| 4 | `energy_model` | Analog crossbar MAC energy (pJ/MAC) vs. this repo's existing CPU/GPU/NPU digital numbers |
| 5 | `dataflow_model` | From-scratch minimal Timeloop/Accelergy-style PPA model: Weight-/Output-/Input-/Row-Stationary dataflows |
| 6 | `systolic_sweep` | PE array design-space sweep on `transformer/`'s real GEMM shapes |
| 7 | `codesign_case_study` | GPTQ (Phase 9) re-derived under analog crossbar constraints, on a real trained transformer weight |
| 8 | `circuit_transient` | RC step-response model: settling time/bandwidth vs. crossbar size |

## Design highlights — how the eight steps chain together

Each step reuses the previous ones' real components rather than building
eight disconnected models:

- **Step 2 injects step 1's noise model** into a real Ohm's-law/KCL
  crossbar simulation, not a separate error term.
- **Step 4's digital-device numbers are reused directly from
  `npu_engine/cost_model`**, not re-guessed, so the analog-vs-digital
  comparison is against numbers this repo already committed to elsewhere.
- **Step 5's dataflow model is reused unmodified by both step 6 (sweeping
  PE size on step 6's own GEMM shapes) and indirectly informs step 7's
  precision framing.**
- **Step 7 reuses Phase 9's real `GptqQuantizer` AND step 1's device model
  together**, on a real trained transformer weight — the co-design case
  study is a pipeline of already-tested components, not new math.
- **Step 8 adds back a cost step 2 deliberately left out** (instantaneous
  reads), completing the picture step 2 started.

## Real findings, not assumed conclusions

- **Precision drives crossbar accuracy far more than crossbar size does**
  (step 2): relative RMSE goes 23.7% -> 1.8% across a 4-to-64-level
  precision sweep, but stays roughly flat (5-7%) across an 8x8-to-64x64
  size sweep — a genuinely counter-intuitive result (signal and analog
  noise both scale as `sqrt(M)` for random weights, so size alone doesn't
  average anything out).
- **A real bug caught by a test, not inspection** (step 1): the endurance
  model's first version re-rolled a Bernoulli stuck-check on every write,
  compounding into a 50%-stuck-at-100-writes result 100x below the rated
  endurance; fixed to 4.0% via a per-cell fixed failure-threshold
  percentile compared against a cumulative curve.
- **RRAM wins an illustrative NVM comparison despite not leading on any
  single axis** (step 3): STT-MRAM has ~1000x its endurance, SRAM-CIM
  1000x endurance + 20x lower write energy — RRAM wins because it has no
  hard disqualifier (unlike SRAM-CIM's zero retention or STT-MRAM's
  near-binary 2 analog levels).
- **A genuinely counter-intuitive energy result** (step 4): at 32-level
  precision, realistic (ADC-inclusive) analog crossbar energy is ~6x
  WORSE than a purpose-built digital NPU, though still ~1.7x better than
  GPU and ~25x better than CPU — analog beats general-purpose digital
  compute but not necessarily a power-efficient fixed-function digital
  accelerator, echoing a real debate in the compute-in-memory literature.
- **Row-Stationary's headline property confirmed numerically** (step 5):
  matches Weight-Stationary's minimal weight movement AND Input-
  Stationary's minimal input movement simultaneously — something neither
  achieves alone.
- **An enormous, measured over-provisioning cost** (step 6): averaged
  across 7 real GEMM shapes from `transformer/`'s actual trained config, a
  128x128 array (matching TPU's real MXU tile boundary) achieves 3.79%
  utilization vs. 100% at each shape's best-matched size.
- **A real non-monotonic result at the end-task level, caught by running
  the test** (step 7): analog noise very slightly IMPROVED perplexity on
  a heavily-overfit toy corpus despite always increasing weight RMSE —
  not a bug (RMSE is guaranteed monotonic by construction; perplexity
  near a sharp optimum isn't). Led to replacing an overly strong
  assertion with a defensible one.
- **A third, independent reason bigger crossbars aren't free** (step 8):
  settling time grows quadratically with crossbar size (Elmore-style
  distributed RC line) — 256x slower for a 16x size increase — a
  completely different mechanism from step 2's statistics-based finding,
  both converging on the same conclusion.

See each step's own README for full methodology and captured output;
`analog_engine/DESIGN.md` for the phase-level design rationale.

## Hardware notes

No analog/neuromorphic silicon exists to rent from any cloud provider —
unlike Phase 3/7/8/15, this phase has no "hardware validation" pass to
defer to. Every literature-informed constant (device noise/drift/
endurance in step 1, NVM comparison points in step 3, ADC energy in step
4, DRAM energy in step 5, R/C in step 8) is explicitly labeled as
illustrative/order-of-magnitude, not measured, in its own step's
README — the honest ceiling on what this phase can claim without fab
access.

## Next

Phase 18 (Dynamical Systems, SciML & Physics-Informed Architectures) was
scoped alongside this phase and is being implemented in parallel this same
session — see `sciml/README.md` (once its own wrap-up lands) and
`READING_LIST.md`'s Phase 18 section for progress.
