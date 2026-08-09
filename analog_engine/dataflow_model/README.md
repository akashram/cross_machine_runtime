# dataflow_model

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 17 step 5: a from-scratch, minimal reproduction of what
Timeloop (Parashar et al. 2019) and Accelergy (Wu, Emer & Sze 2019)
compute — given a GEMM workload and a PE-array architecture description,
analytically count data-movement volume per operand and estimate energy,
for each of the four dataflow strategies Sze, Chen, Yang & Emer (2017)
survey: Weight-/Output-/Input-/Row-Stationary.

## Design

- **2-level memory hierarchy** (DRAM ↔ PE-array-local), GEMM
  `C[M,N]=A[M,K]*B[K,N]` tiled onto a `Prows x Pcols` PE array with `K`
  itself tiled into `k_tile_size`-deep passes — this is what makes
  Output-Stationary's real advantage show up numerically: a dataflow that
  doesn't keep partial sums resident across K-tile passes pays a real
  read-back-and-accumulate cost (`2*k_tiles-1` round trips per output
  tile); without K-tiling, that distinction wouldn't appear at all, an
  honestly-disclosed choice.
- **Row-Stationary is a GEMM-adapted simplification of Eyeriss's actual
  conv-specific row mapping** (Chen, Emer & Sze 2016/2017) — rather than
  reproducing Eyeriss's 1D-conv-filter-row reuse pattern (which needs a
  spatial conv dimension a pure GEMM doesn't have), RS here holds a
  K-tile-sized row of BOTH weight and input resident simultaneously,
  reused across the full output tile space before moving to the next
  K-tile. This captures RS's real headline property — near-ideal reuse of
  TWO operands at once, where WS/IS each only achieve that for ONE — at
  the same K-tile partial-sum cost WS/IS pay. A disclosed adaptation, not
  a literal Eyeriss reproduction.
- PE utilization uses TILE-PADDED dimensions, so it directly reproduces
  `tpu_engine/mxu_opt`'s own utilization-cliff phenomenon for GEMM
  dimensions that don't divide evenly into the PE array.

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  GEMM 512x512x512 on a 128x128 PE array, k_tile=128:
  dataflow               weight_moved    input_moved   output_moved    pe_util    energy_pJ
  Weight-Stationary            262144        1048576        1835008      1.000   62914560.0
  Output-Stationary           1048576        1048576         262144      1.000   47185920.0
  Input-Stationary            1048576         262144        1835008      1.000   62914560.0
  Row-Stationary               262144         262144        1835008      1.000   47185920.0
PASS  Weight-Stationary achieves the minimum weight movement among all four dataflows
PASS  Input-Stationary achieves the minimum input movement among all four dataflows
PASS  Output-Stationary achieves the minimum output/partial-sum movement among all four dataflows
PASS  Row-Stationary matches Weight-Stationary's (minimal) weight movement
PASS  Row-Stationary matches Input-Stationary's (minimal) input movement -- SIMULTANEOUSLY with the weight result above, RS's real headline property no single-operand-stationary dataflow achieves
PASS  PE utilization is exactly 1.0 when GEMM dimensions divide evenly into the PE array (no padding waste)

  misaligned M=513 (not a multiple of 128): pe_utilization=0.8016
PASS  PE utilization drops below 1.0 for GEMM dimensions that don't divide evenly into the PE array -- the same utilization-cliff phenomenon tpu_engine/mxu_opt measured on real TPU MXU alignment
```

## Findings

- **Row-Stationary's headline property is real in the numbers, not just
  asserted**: it achieves `262144` weight elements moved (tied with WS's
  minimum) AND `262144` input elements moved (tied with IS's minimum)
  SIMULTANEOUSLY — something neither WS nor IS achieves alone (each wins
  on exactly one operand, loses on the other). This is the actual reason
  Eyeriss's row-stationary dataflow is a genuine third design point, not
  just "WS and IS averaged."
- **A real, disclosed artifact of this symmetric workload**: at
  `M=K=N=512` with a square PE array, Weight-Stationary and Input-
  Stationary land on IDENTICAL total energy (`62,914,560 pJ` each), and
  Row-Stationary and Output-Stationary also land on an identical total
  (`47,185,920 pJ` each) — RS gets WS+IS's combined weight/input minimum
  but pays WS/IS's output cost, while OS gets the opposite trade, and at
  this specific symmetric shape the two trades net out equal. A workload
  with `M != N` would break this tie (WS and IS would diverge from each
  other, and RS/OS would very likely diverge from each other too) — worth
  stating as a property of the CHOSEN workload's symmetry, not a general
  law that RS always ties OS.
- **Misaligned utilization reproduces `tpu_engine/mxu_opt`'s own
  finding**, from a completely independent model built for a different
  hardware target: `M=513` (one row past a 128-boundary) drops PE
  utilization to `0.8016` — nearly 20% of the array sits idle to cover a
  single extra row's worth of tile padding. Same qualitative phenomenon
  (small misalignment, large utilization cost near a tile boundary) two
  independently-built models in this repo both surface.

## Hardware notes
None — pure CPU. The `kDramPjPerElement` energy constant is an
illustrative literature-order-of-magnitude figure (no fab access to
measure real DRAM access energy directly).
