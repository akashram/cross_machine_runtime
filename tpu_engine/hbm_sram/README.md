# hbm_sram

**Status: analytical model done and run locally (`hbm_sram_model.cpp`); real
DMA-overlap measurement TODO — no TPU on this Mac.**

## What this measures

PLAN.md Phase 8 step 6: explicit HBM<->SRAM (VMEM) data movement scheduling.
A TPU has no hardware cache — every tile the MXU consumes crosses HBM to the
on-chip VMEM scratchpad via a DMA the compiler (XLA) scheduled explicitly at
compile time, not a runtime cache-fill decision. This step documents and
models that scheduling; it does not add new IR (the `runtime` dialect's
`transfer` op, from `compiler/dialect`, already covers explicit device-side
data movement — see stablehlo_lower's design table for how it lowers, or
rather deliberately doesn't, on the TPU path, since XLA owns HBM/VMEM
scheduling internally once StableHLO reaches it).

## Design

`hbm_sram_model.cpp`: analytical (not simulated — no `std::chrono`, since
there's no real transfer to time without a TPU) double-buffering model.
For a given matmul tile shape, computes MXU compute time (from peak bf16
TFLOPS) and HBM transfer time (from peak HBM GB/s, both the same published
TPU v4 specs `tpu_benchmarks/` uses) and reports which one bounds
steady-state double-buffered throughput, and the resulting overlap
efficiency (useful MXU time / wall time).

Build/run directly, no CMake, same convention as `layout_opt`:
`clang++ -O2 -std=c++17 hbm_sram_model.cpp -o hbm_sram_model && ./hbm_sram_model`

## Results (captured 2026-07-27, Apple clang 14, this Mac)

```
tile shape                      compute(us) transfer(us)      bound     overlap eff.
small tile 128x128x128                 0.02         0.05   transfer            27.9%
medium tile 512x512x512                0.98         0.87    compute           100.0%
large tile 2048x2048x2048             62.47        13.98    compute           100.0%
tall-skinny 4096x128x4096             15.62        28.84   transfer            54.2%
wide 128x4096x128                      0.49         0.90   transfer            54.2%
```

## Findings

- Confirms the arithmetic-intensity argument analytically: a cubic tile
  (m=n=k) crosses from transfer-bound to compute-bound as size grows
  (128^3 is transfer-bound at 27.9% overlap efficiency, 512^3 and 2048^3
  are already fully compute-bound) because FLOPs scale as size^3 while
  transferred bytes scale as size^2 — larger cubic tiles do more work per
  byte moved.
- Tall-skinny and wide tiles stay transfer-bound (54.2% overlap
  efficiency) *regardless* of how large the other two dimensions grow,
  since one small dimension caps arithmetic intensity — this is the same
  structural finding `layout_opt`'s small-batch case hits from a different
  angle (batch=1 decode: M pinned small), now shown to also cost overlap
  efficiency, not just MXU tile-padding utilization.
- Practical implication: XLA's tiling heuristics favoring roughly-cubic
  tiles (when VMEM capacity allows) isn't an arbitrary choice — it's the
  direct consequence of this compute-vs-transfer crossover, and it's a
  claim this model makes independent of whether the specific TFLOPS/GB/s
  constants used are exactly right (see Hardware notes).

## Hardware notes
- Required for calibration: a TPU VM's real achieved HBM bandwidth and MXU
  TFLOPS (`tpu_benchmarks/`'s step 2 scripts) in place of the published
  peak-spec constants this model uses — achieved throughput is normally
  below peak, which would shift the compute/transfer crossover point
  measured here.
