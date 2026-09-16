# Phase 20: Quantum Computing & Hybrid Quantum-Classical Compute

**Status: CODE COMPLETE (10/10 steps), 2026-09-16.** Unlike Phase 17 (no
rentable analog/neuromorphic silicon exists at all) or Phase 3/7/8/15
(real hardware exists but is toolchain/rental-gated), quantum computing
sits in a third position: no quantum hardware exists locally, but
classical simulation of a modest qubit count is not a stand-in — it is
the exact, correct physics, just without noise or the scale where
classical simulation becomes intractable. So 9 of 10 steps are real,
run-for-real code (all actually compiled, run, and captured with real
output below); only step 7 (PennyLane) stays genuinely unrun, since the
user declined installing PennyLane this session.

## Overview

Scoped 2026-09-12, closing a gap distinct from Phases 17-19's JD-gap-
analysis origin: this repo had zero quantum computing content at all, a
fully separate compute paradigm from everything in Phases 1-19 (including
Phase 17's analog compute, which is resistive/neuromorphic, not quantum).

## Steps

| # | Directory | What | Status |
|---|-----------|------|--------|
| 1 | `state_vector` | Hand-rolled state-vector simulator: gates, measurement, projection | Run locally |
| 2 | `algorithms` | Deutsch-Jozsa, Grover's search, QFT | Run locally |
| 3 | `noise_model` | Depolarizing + amplitude-damping channels via Monte Carlo wavefunction unraveling | Run locally |
| 4 | `qec` | 3-qubit bit-flip repetition code, real syndrome-measurement circuit | Run locally |
| 5 | `vqe` | Variational Quantum Eigensolver, parameter-shift-rule gradient descent | Run locally |
| 6 | `qaoa` | QAOA on MaxCut, checked against classical brute force | Run locally |
| 7 | `pennylane_native` | PennyLane framework-native VQE/QAOA reimplementation | **Unrun** — no PennyLane install |
| 8 | `cv_photonic` | Continuous-variable/photonic primitives, hand-rolled Gaussian states | Run locally |
| 9 | `qpu_backend` | QPU registered in `inference_serving::ServingRouter`; real shot-sampling call path | Run locally |
| 10 | `cost_model` | Cloud QPU provider landscape (IBM/AWS Braket/Azure/Xanadu), literature-grounded | Run locally |

## Design highlights — how the ten steps chain together

- **Steps 2-9 all build on step 1's simulator unmodified.** No step
  re-implements gate application, measurement, or projection — `qec`,
  `vqe`, `qaoa`, and `qpu_backend` all call `state_vector.h`'s primitives
  directly, and step 1 itself grew two new primitives (`renormalize()`,
  `measure_qubit()`) exactly when a later step's physics genuinely needed
  them (steps 3 and 4 respectively), not speculatively up front.
- **Step 3's noise model feeds step 4's QEC demonstration directly** —
  `qec_inject_depolarizing_errors` reuses `apply_depolarizing` unmodified
  to show the bit-flip code's protection degrading under a noise type it
  wasn't designed for, rather than inventing a second noise
  implementation.
- **Step 9 reuses Phase 15's exact `ServingRouter` integration pattern**
  (`Backend::QPU`, `available=false` + reason string, appended to the
  fallback priority order) — additive, zero regressions to
  `serving_router_test.cpp`'s existing 12 assertions.
- **Step 10 explicitly informs the eventual hardware-validation choice**:
  its written discussion names Xanadu Cloud as the most directly relevant
  provider specifically BECAUSE it reaches the same software stack (step
  7's PennyLane) and formalism (step 8's Gaussian states) this phase
  already built, not a generic "best QPU" ranking.
- **Step 7 (PennyLane) is real, complete, syntactically-valid code, kept
  structurally identical to steps 5/6's hand-rolled circuits specifically
  so the eventual run is a direct comparison**, not a second
  from-scratch implementation.

## Real findings, not assumed conclusions

- **Grover's periodicity, caught by a brute-force test design mistake**
  (step 2): a wide-range brute-force search over iteration counts
  initially "failed" the closed-form optimal-iterations formula — not a
  bug, but real periodicity in success probability (`sin^2((2k+1)*theta)`
  repeats every `pi/(2*theta)` iterations), fixed by windowing the search
  around the formula's own prediction.
- **A real power-iteration normalization bug, caught by a hand-derivable
  closed-form check** (step 5): `std::norm()` on an already-real complex
  scalar was squaring an already-squared quantity, giving `-0.601` instead
  of the exact `-2.0` answer for `H=Z0+Z1` — caught immediately because a
  closed-form ground truth existed, fixed, then reused to validate VQE on
  a Hamiltonian with no simple closed form.
- **A genuine ansatz-expressibility ceiling, not an optimizer artifact**
  (step 5): depth-1/2 VQE plateaus at the IDENTICAL suboptimal energy
  across 8 independent random restarts on a 3-qubit TFIM Hamiltonian,
  while depth-4 reaches the exact ground energy from every restart —
  ruling out "bad luck" as the explanation.
- **A real learning-rate instability, documented via an explicit sweep**
  (step 6): QAOA's first `lr=0.3` attempt oscillated below its own
  starting point; a `{0.3,0.1,0.05,0.02,0.01}` sweep showed the
  instability threshold directly, and `p=2` then reaches the EXACT
  classical MaxCut optimum on a 5-node odd cycle.
- **A degraded-but-real protection result, disclosed not hidden** (step
  4): the bit-flip code's logical error rate tracks the analytic
  `3p^2-2p^3` formula closely under pure bit-flip noise (beating physical
  error rate by ~10x at small `p`), but protects far worse under general
  depolarizing noise (X+Y+Z) — the code only targets one error type, and
  the measurement shows exactly how much that limitation costs.
- **Total photon number conservation, exact to `1e-9`** (step 8):
  confirms `beamsplitter_local_matrix` is genuinely passive/
  energy-conserving, not just symplectic in the abstract — squeezing is
  ALSO symplectic and clearly does NOT conserve photon number, so this is
  a real distinguishing check, not redundant with the symplectic-property
  check both gates pass.

See each step's own README for full methodology and captured output;
`quantum_engine/DESIGN.md` for the phase-level design rationale.

## Hardware notes

Unlike Phase 17, quantum hardware access genuinely exists (IBM Quantum's
free tier, AWS Braket, Azure Quantum, Xanadu Cloud) — see step 10's
`cost_model/README.md` for the full landscape and which provider fits
which step. What's simulated here at small qubit counts is EXACT, not
approximate; the gap to real hardware is noise realism and scale, not
correctness. Step 7 (PennyLane) is the one step genuinely blocked on a
local install decision rather than cloud hardware access — see
`pennylane_native/README.md`.
