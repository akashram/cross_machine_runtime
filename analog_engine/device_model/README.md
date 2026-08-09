# device_model

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 17 step 1: a parameterized RRAM-like conductance-cell
non-ideality model — read/write noise, conductance drift over time, and
endurance-linked cell failure — the device physics step 2's crossbar
matrix-vector-multiply simulation injects, and step 3's NVM comparison
table is grounded in.

No RRAM hardware exists to characterize directly here (no fab access, same
disclosed limitation as this repo's other hardware-gated phases), so the
noise/drift/endurance magnitudes are literature-informed order-of-magnitude
values from Yu, S. (2018), *"Neuro-Inspired Computing with Emerging
Nonvolatile Memory"* (Proceedings of the IEEE) — not measurements of a real
device. That distinction is the point of this step: model the KNOWN
qualitative behaviors with plausible magnitudes, honestly labeled, rather
than assume a digital-precision abstraction is what analog compute-in-memory
hardware actually provides.

## Design

- `ConductanceCell` tracks one simulated cell's actual conductance,
  write-cycle count, and stuck/failed state.
- **Write noise > read noise, deliberately.** `write()` programs toward a
  target conductance level plus Gaussian noise (cycle-to-cycle
  variability, the dominant real MAC-error source in a crossbar); `read()`
  adds a strictly smaller Gaussian perturbation (thermal/measurement
  noise, not stochastic filament switching) — `read_noise_frac_of_write <
  1.0` enforces this by construction, and the test verifies it empirically
  rather than trusting the parameter.
- **Conductance drift** follows the standard empirical power law
  `G(t) = G_written * (t/t_ref)^(-nu)` (Yu 2018, Sec. III) — checked
  against a `nu=0` control run to isolate drift specifically as the cause
  of decay, not some other time-dependent effect.
- **Endurance-linked failure is a per-cell CUMULATIVE percentile check,
  not a per-write coin flip** (see Findings — this is where a real bug
  was caught and fixed): each cell draws one random failure-threshold
  percentile at construction; on every write, if the logistic cumulative-
  failure curve evaluated at the current cycle count exceeds that cell's
  percentile, the cell becomes permanently stuck (its conductance frozen
  wherever it was — real devices don't announce failure, they just stop
  responding).

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  stddev across independent writes = 1.6322 | stddev across repeated reads of one write = 0.3422
PASS  read() perturbs a cell's measured conductance far less than write() does (read noise < write noise, as physically expected)
  nearest-level round-trip accuracy at num_levels=8: 96.0% (384/400)
PASS  writing a level then reading immediately recovers it via nearest-level classification with >95% accuracy at 8 well-separated levels
  mean conductance: t=t_ref -> 80.2238 | t=1e6*t_ref -> 26.5433 (nu=0.08)
PASS  conductance drifts DOWNWARD over time under the G(t) = G0*(t/t_ref)^(-nu) power-law model (nu > 0)
  with drift_nu=0: mean conductance at t=1e6*t_ref -> 80.2238 (should match early-time mean, no decay)
PASS  drift_nu=0 leaves conductance essentially unchanged over the same time span (isolates drift as the cause above, not some other time-dependent effect)
  stuck fraction after 100 writes (100x below rated endurance) = 0.040
  stuck fraction after 100000 writes (100x above rated endurance) = 1.000
PASS  cells written far below their rated endurance are almost never stuck
PASS  cells written far beyond their rated endurance become stuck with high probability
PASS
```

## Findings

- **A real bug, caught by actually running the test, not by inspection.**
  The first version of the endurance model rolled a fresh Bernoulli
  stuck-check on EVERY write call using the logistic curve evaluated at
  the current cycle count. That's wrong: it turns a single
  population-level failure probability into dozens of near-independent
  chances to fail before reaching a given cycle count, compounding into a
  wildly inflated stuck rate. At 100 writes (100x below the rated 1000-
  cycle endurance in the test), this produced a 50% stuck fraction —
  nonsensical for a cell nowhere near its rated endurance. Fixed by
  drawing one failure-threshold percentile per cell at construction and
  comparing the CUMULATIVE failure curve against that fixed percentile on
  each write instead of re-rolling every time; the same test now reports
  4.0% stuck at 100 writes (below the <5% bar) and 100% stuck at 100,000
  writes (above the >90% bar) — a coherent population-reliability curve
  instead of compounding hazard.
- Read noise came out ~5x smaller than write-to-write spread
  (stddev 0.34 vs. 1.63) at the default parameters, consistent with the
  physical claim that reading a cell disturbs it far less than programming
  it.
- Level round-trip accuracy at 8 well-separated levels is 96% — high, but
  not perfect, at default write-noise settings; this is the exact
  precision-vs-accuracy tradeoff step 2's crossbar MAC simulation will
  measure as a function of `num_levels`, not a separate finding.

## Hardware notes
None — pure CPU. No analog/neuromorphic silicon exists to characterize
directly (see PLAN.md's Phase 17 hardware-access note); this model's
constants are literature-informed, not measured.
