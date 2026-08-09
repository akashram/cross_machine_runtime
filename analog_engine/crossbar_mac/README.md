# crossbar_mac

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 17 step 2: a resistive crossbar's analog matrix-vector
multiply (Ohm's law `I = G*V` per cell, Kirchhoff's current law summing
currents into a row "for free," on the wire, not as a separate compute
step), with step 1's device non-idealities injected — measured MAC
accuracy vs. the exact digital reference as a function of bit-precision
and crossbar size.

## Design

- **Signed weights via a differential cell pair**, the standard real
  technique (ISAAC, PRIME — see READING_LIST.md's Phase 17 step 2 entry),
  not a simplification unique to this repo: a physical conductance can't
  be negative, so each weight `w_ij` is encoded as `(G_pos, G_neg)`, one
  programmed to `|w_ij|`'s quantized level and the other left at baseline,
  so that `(G_pos - G_neg) = sign(w_ij) * |w_ij| * (g_max - g_min)` exactly
  in the noiseless limit. Input voltages, unlike conductances, can be
  signed directly — no encoding needed on that side.
- `Crossbar::multiply()` reads every cell (with step 1's read noise/drift
  applied) and sums per row — the KCL summation the physical wire does for
  free is what this loop MEASURES, not what it computes.
- Three tests: basic accuracy at a representative size/precision, a
  precision sweep (`num_levels`) at FIXED crossbar size and FIXED random
  `W`/`x` (isolating precision's effect alone), and a crossbar-size sweep
  at fixed precision (measuring the real relationship, not assuming
  bigger-averages-out-noise or the reverse).

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  16x16 crossbar, num_levels=32: RMSE=0.0438, relative RMSE=3.63%
PASS  16x16 crossbar at 32 levels achieves <20% relative RMSE vs. exact digital matvec
  precision sweep (fixed 16x16 W/x, write_noise_frac_of_level=0.25):
    num_levels=  4 -> relative RMSE=23.71%
    num_levels=  8 -> relative RMSE=15.63%
    num_levels= 16 -> relative RMSE=8.02%
    num_levels= 32 -> relative RMSE=3.75%
    num_levels= 64 -> relative RMSE=1.79%
PASS  relative RMSE at 4 levels is worse than at 64 levels (more precision helps, measured across the full sweep)
    monotonic-or-flat improvements: 4/4 consecutive steps
PASS  accuracy improves (or holds) at nearly every step of the precision sweep, not just end-to-end
  crossbar-size sweep (fixed num_levels=16, fresh random W/x per size):
      8x8   -> relative RMSE=5.72%
     16x16  -> relative RMSE=6.65%
     32x32  -> relative RMSE=4.34%
     64x64  -> relative RMSE=5.06%
PASS  crossbar-size sweep completed and reported (see below -- not assumed in either direction)
```

## Findings

- **Precision sweep: clean, monotonic, and roughly halves per level-count
  doubling** — 23.71% -> 15.63% -> 8.02% -> 3.75% -> 1.79% relative RMSE
  as `num_levels` goes 4 -> 8 -> 16 -> 32 -> 64. This tracks the model
  directly: `write_noise_frac_of_level` is a fixed FRACTION of one level's
  spacing, and that spacing halves each time `num_levels` doubles, so both
  the absolute write noise per cell and the quantization error shrink
  together — a coherent, expected result, not a coincidence of the random
  seed (same `W`/`x` used at every precision in the sweep).
- **Crossbar-size sweep: a genuinely counter-intuitive, real finding.**
  Relative RMSE stayed roughly FLAT across 8x8 -> 64x64 (5.7%, 6.7%, 4.3%,
  5.1%) — it did NOT improve as the crossbar got bigger, despite the
  intuitive expectation that averaging over more terms should cancel more
  noise. The reason: for random (mean-zero) weights and inputs, BOTH the
  true signal `y_i = sum_j w_ij*x_j` and the accumulated analog noise are
  sums of `M` independent zero-mean terms, so both grow as `sqrt(M)` (a
  random walk) rather than the signal growing linearly while noise
  averages out — their ratio (the thing `relative_rmse` measures) stays
  roughly constant with crossbar size instead of improving. This is
  measured here, not assumed either direction going in — the size sweep
  test intentionally asserts nothing about which way the trend should go,
  only that the measurement itself completed and got reported.
- Together, these two findings say something specific and testable about
  step 5/6's PPA modeling: crossbar SIZE is not, by itself, an accuracy
  lever for random workloads — bit PRECISION is. Scaling a crossbar up
  buys density and throughput, not free noise-averaging.

## Hardware notes
None — pure CPU. No analog/neuromorphic silicon exists to characterize
directly (see PLAN.md's Phase 17 hardware-access note).
