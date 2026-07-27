# cost_model

**Status: code-complete AND locally run — no TPU/GPU/JAX dependency, same
convention as `compiler/cost_model` (Phase 4 step 13, the one Phase 4
component that doesn't need MLIR). Calibration constants are public
list-price/spec-sheet snapshots, not measurements — same explicit caveat
`compiler/cost_model/README.md` states for its own numbers.**

## What this measures

PLAN.md Phase 8 step 12: $/FLOP and FLOPS/Watt for TPU v4 vs. A100 vs.
H100, and when to choose each.

## Design

`tpu_cost_model.cpp`: three `DeviceSpec` constants (peak dense bf16/
tensor-core TFLOPS, TDP watts, on-demand $/hr, price source), derives
TFLOPS/Watt and $/PFLOP-hour (cost to rent one PFLOP/s of peak compute for
an hour) for each. Peak-TFLOPS constants for A100/TPU v4 are the actual
ML-relevant tensor-core/MXU peaks (312 / 275 TFLOPS bf16) — a different,
larger number than `compiler/cost_model/CostModel.cpp`'s existing
`DeviceCost` table uses for A100 (19.5 TFLOPS, the *non*-tensor-core FP32
path that estimator was calibrated against, predating dtype-aware costing
per that file's own header TODO) — documented in this file's header so the
two don't get conflated as disagreeing about the same number.

Build/run directly, no CMake, same convention as `layout_opt`/`hbm_sram`:
`clang++ -O2 -std=c++17 tpu_cost_model.cpp -o tpu_cost_model && ./tpu_cost_model`

## Results (captured 2026-07-27, Apple clang 14, this Mac)

```
device             TFLOPS(bf16)   TDP(W)       $/hr      TFLOPS/Watt         $/PFLOP-hr
TPU v4                    275.0      192       3.22            1.432              11.71
A100 80GB SXM4            312.0      400       3.67            0.780              11.76
H100 80GB SXM5            989.0      700      11.06            1.413              11.18
```

Price sources (snapshot, verify before procurement — cloud on-demand
pricing changes often and varies by region/commitment):
- TPU v4: GCP `v4` on-demand, per chip (or free via TRC)
- A100 80GB SXM4: GCP `a2-ultragpu-1g` on-demand, per GPU
- H100 80GB SXM5: GCP `a3-highgpu-8g` on-demand / 8, per GPU

## Findings

- On raw $/PFLOP-hour, all three land within ~5% of each other
  (~$11-12) — at peak, dense-bf16, on-demand list price, none of the three
  has a decisive cost edge over the others; the "which to choose" decision
  in practice hinges almost entirely on whether a given workload can
  actually *reach* peak on each device (layout_opt's MXU-alignment
  ceiling, tensor-core tile-shape equivalents on A100/H100), not this
  table's peak-vs-peak ranking.
- TPU v4 and H100 are close on FLOPS/Watt (1.43 vs. 1.41), both well ahead
  of A100 (0.78) — A100 is the oldest of the three generations, so this
  matches the expected trend of newer silicon buying more compute per
  watt, not a TPU-specific advantage.
- Practical guidance this repo can act on: TPU v4 and H100 are close
  enough on both axes that the deciding factor for a given workload in
  this repo should be whichever one that workload's op mix can actually
  reach a high utilization ceiling on — a workload that hits
  `layout_opt`'s cliff badly on TPU (e.g. batch=1 decode, per that step's
  finding) may do meaningfully better on $/useful-FLOP on H100 even though
  this table's peak numbers put them nearly level, and vice versa for a
  workload shaped exactly to TPU's 128x128 tiles.

## Hardware notes
None — this step has no hardware dependency, unlike the rest of Phase 8.
