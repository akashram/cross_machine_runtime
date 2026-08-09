# energy_model

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 17 step 4: analog crossbar MAC energy (pJ/MAC) vs. this
repo's EXISTING digital device numbers (CPU/GPU/NPU, reused directly from
`npu_engine/cost_model/npu_cost_model.cpp`'s TOPS/W constants, not
re-guessed) — a real quantitative "where and by how much does analog win
on energy" comparison.

## Design

- Digital pJ/MAC is derived from the SAME TOPS/W constants
  `npu_engine/cost_model` already established for CPU/GPU/NPU (`pJ/MAC =
  2 / (TOPS/W)`, since 1 MAC = 2 ops) — reused, not a second independent
  guess at the same devices.
- Analog is split into two numbers on purpose: **pure compute** (the
  crossbar's raw Ohm's-law multiply-accumulate alone, ~femtojoule scale)
  and **realistic (ADC-inclusive)** (adding the cost of reading the analog
  result back into the digital domain, which the compute-in-memory
  literature — Yu 2018 — documents as often DOMINATING a real crossbar
  accelerator's total energy). Both are illustrative literature-order-of-
  magnitude points, honestly labeled as such (no fab access).
- ADC energy is modeled as scaling with bits of resolution
  (`log2(num_levels)`) — the direct connection to step 2's own finding
  that `num_levels` is what drives crossbar MAC accuracy: this step shows
  the other side of that same knob, precision costs real ADC energy, not
  a free lunch.

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  digital devices (pJ/MAC, derived from npu_engine/cost_model's TOPS/W constants):
    CPU (AVX-512 VNNI server)    0.053 TOPS/W -> 37.5000 pJ/MAC
    GPU (A100, dense INT8)       0.780 TOPS/W -> 2.5641 pJ/MAC
    NPU (Apple ANE, repr.)       7.900 TOPS/W -> 0.2532 pJ/MAC

  analog crossbar (illustrative, literature order-of-magnitude, no fab access):
    pure compute (Ohm's-law MAC alone): 0.0050 pJ/MAC
    realistic (ADC-inclusive) by precision:
      num_levels=  4 (2-bit): 0.6050 pJ/MAC
      num_levels=  8 (3-bit): 0.9050 pJ/MAC
      num_levels= 16 (4-bit): 1.2050 pJ/MAC
      num_levels= 32 (5-bit): 1.5050 pJ/MAC
      num_levels= 64 (6-bit): 1.8050 pJ/MAC
PASS  pure-compute analog MAC energy is more than 10x cheaper than the ADC-inclusive realistic figure, even at the COARSEST precision -- ADC overhead dominates, the real reason isolated 'analog is free' claims mislead
PASS  realistic (ADC-inclusive) analog MAC energy strictly increases with num_levels -- precision has a real energy cost, connecting directly to step 2's finding that precision drives accuracy

  at num_levels=32 (5-bit, step 2's representative precision): realistic analog=1.5050 pJ/MAC vs. NPU=0.2532, GPU=2.5641, CPU=37.5000
PASS  realistic (ADC-inclusive) analog crossbar beats both GPU and CPU on pJ/MAC at 32-level precision -- a real margin, not just the misleading pure-compute number
```

## Findings

- **ADC overhead genuinely dominates, by a wide margin**: the "pure
  compute" figure (`0.005 pJ/MAC`) is 121x cheaper than the realistic
  figure at the coarsest precision tested (`0.605 pJ/MAC` at 4 levels) and
  361x cheaper at 64 levels (`1.805 pJ/MAC`). Any comparison quoting
  analog compute-in-memory's "free multiply-accumulate" number in
  isolation, without the ADC readout cost, is quoting a number that isn't
  representative of a deployable accelerator's actual energy.
- **A genuinely counter-intuitive result at this model's parameters: a
  well-optimized fixed-function digital NPU beats the realistic
  (ADC-inclusive) analog crossbar on pJ/MAC.** At 32-level (5-bit)
  precision — step 2's representative precision point — realistic analog
  costs `1.505 pJ/MAC` against the NPU's `0.253 pJ/MAC`, nearly 6x
  WORSE, not better. Analog does still clearly beat GPU (`2.564 pJ/MAC`,
  analog ~1.7x better) and CPU (`37.5 pJ/MAC`, analog ~25x better) — the
  "analog wins on energy" story holds against general-purpose digital
  compute, but not necessarily against a purpose-built, already highly
  power-efficient fixed-function digital accelerator. This directly
  echoes a real, active debate in the compute-in-memory literature (full
  peripheral-circuit accounting often erodes or reverses analog's
  headline energy advantage) rather than a repo-specific artifact — but
  the specific crossover point IS sensitive to the illustrative ADC-per-
  bit constant chosen here (`0.3 pJ/bit`), which is explicitly not a
  measured or single-paper-cited number; a more/less aggressive ADC
  design would shift where NPU and analog cross, not the qualitative
  shape of the finding (ADC cost scales with precision, digital doesn't
  pay that particular tax the same way).
- Precision's energy cost is exactly linear in bits (by construction of
  the model — `0.3 pJ` per additional bit of resolution) but the
  ACCURACY benefit from step 2 is not linear in the same variable (step 2
  measured diminishing but still substantial returns: 23.7% -> 15.6% ->
  8.0% -> 3.75% -> 1.79% relative RMSE across the same 4/8/16/32/64-level
  sweep) — so the accuracy-per-additional-ADC-pJ tradeoff genuinely gets
  worse at higher precision, a real, quantifiable design tension between
  these two steps' findings.

## Hardware notes
None — pure CPU. No fab access exists to measure real crossbar ADC energy
directly (see PLAN.md's Phase 17 hardware-access note); the digital-device
numbers are likewise the same spec-sheet-derived figures
`npu_engine/cost_model` already uses, not new measurements.
