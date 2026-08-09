# nvm_comparison

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 17 step 3: a non-volatile-memory device-class comparison
(RRAM, PCM, STT-MRAM, SRAM-CIM) for use as an analog compute-in-memory
substrate — literature-grounded, honestly labeled as such rather than
presented as measured (no fab access exists to characterize any of these
devices directly).

## Design

- Every numeric field is a **representative point drawn from a cited
  order-of-magnitude range** in Yu, S. (2018), *"Neuro-Inspired Computing
  with Emerging Nonvolatile Memory"* — the same source step 1's device
  model draws its noise/drift/endurance constants from. Each table entry's
  comment states the cited range and the qualitative reasoning behind the
  chosen point, not just the number.
- **SRAM-CIM is included on purpose as the non-nonvolatile baseline** — its
  retention is 0 by construction (it's volatile), which is the entire
  reason NVM-based analog compute-in-memory is worth the engineering cost
  in the first place.
- The composite figure-of-merit is an explicitly **illustrative** geometric
  mean over five normalized axes (endurance, retention, write energy,
  density, analog levels) — documented in the header as not a validated
  accelerator-design cost model, just a reproducible way to combine five
  genuinely different units into one ranking. A geometric mean means one
  zero (SRAM-CIM's retention) drives the whole score to zero regardless of
  how good the other four axes are — a hard disqualifier the other axes
  can't average away, matching how volatility actually behaves as a design
  constraint.

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  device        endurance  retention   write_pJ    density   levels
  RRAM            1.0e+07       10.0       10.0        8.0       32
  PCM             1.0e+07       10.0      100.0        8.0       16
  STT-MRAM        1.0e+13       10.0        5.0        4.0        2
  SRAM-CIM        1.0e+16        0.0        0.5        1.0       16

  figure-of-merit ranking (analog compute-in-memory suitability, illustrative composite score):
    RRAM       score=0.4656
    STT-MRAM   score=0.3026
    PCM        score=0.2557
    SRAM-CIM   score=0.0000
PASS  SRAM-CIM's zero retention (volatility) drives its composite score to exactly 0.0 via the geometric mean, regardless of its other good axes
PASS  STT-MRAM has strictly fewer max_analog_levels than every other device class -- the real structural binary-favoring tradeoff of magnetic storage
PASS  the top-ranked device by composite score is NOT the volatile baseline (confirms the ranking isn't dominated by SRAM-CIM's best-in-class energy/endurance alone)
```

## Findings

- **RRAM ranks first despite not being best-in-class on any single axis**
  — STT-MRAM has ~1000x RRAM's endurance and half its write energy;
  SRAM-CIM has 1000x RRAM's endurance and 20x lower write energy. RRAM
  wins the composite ranking because it's the only device with no hard
  disqualifier: unlike SRAM-CIM (zero retention) and STT-MRAM (2 analog
  levels — effectively binary), RRAM is simultaneously non-volatile AND
  supports many analog levels, the two properties analog compute-in-memory
  needs together, not separately.
- **STT-MRAM's fundamental limitation for this use case is structural, not
  a parameter that better engineering fixes**: two stable magnetic states
  make genuine multi-level analog operation far harder than RRAM/PCM's
  continuous resistance range. This is the real reason STT-MRAM, despite
  being the most mature/production-proven NVM technology of the four for
  DIGITAL applications, isn't the crossbar substrate ISAAC/PRIME (step 2)
  actually use.
- **This connects directly to step 2's own finding**: step 2 measured that
  crossbar precision (`num_levels`) drives MAC accuracy far more than
  crossbar size does. This step's data shows WHY that matters for device
  selection — STT-MRAM's endurance/energy advantages are irrelevant to
  analog compute-in-memory suitability if its `max_analog_levels=2` caps
  MAC accuracy the way step 2 showed `num_levels=4` does (23.7% relative
  RMSE, the worst point measured in that sweep).

## Hardware notes
None — pure CPU. No fab access exists to characterize any of these device
classes directly; every number is literature-informed, not measured (see
PLAN.md's Phase 17 hardware-access note).
