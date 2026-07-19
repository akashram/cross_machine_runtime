# Device Cost Model

**Status: code-complete AND locally run — this is the one Phase 4 component
with no MLIR dependency. Calibration constants are still spec-sheet
placeholders pending Phases 2/3/7/8 benchmarks.**

## What this measures
Cost model for each device: FLOPS/sec, memory bandwidth, launch overhead,
transfer bandwidth, combined via a roofline-style `max(compute, memory) +
launch_overhead` estimator. Used by the placement pass (step 9) and the
auto-sharding pass (step 10).

## Design
`estimate_us()` and `get_device_cost()` in `CostModel.cpp` — no MLIR types
anywhere in this component, so unlike the rest of Phase 4 it isn't gated
behind `MLIR_DIR` (see root `CMakeLists.txt`). Per-op FLOP/byte formulas
for Matmul/Conv/Elementwise/Reduce/Transfer are documented inline; `Conv`
reduces to a matmul-shaped cost via an im2col-equivalent dimension packing
convention (documented in `CostModel.h`'s `Shape` comment).

## Sanity-run output (Mac, 2026-07-19)

Compiled and run directly (`clang++ -std=c++20 -O2 -Wall -Wextra`, zero
warnings) — this validates the roofline arithmetic and device-dispatch
logic, **not** the calibration constants below, which are public spec-sheet
peaks (A100 FP32/HBM2e, VU9P DSP-slice estimate, TPU v4 bf16/HBM, a generic
AVX-512 server), not measurements:

```
workload                                 CPU (us)         GPU (us)        FPGA (us)         TPU (us)
matmul 4096x4096x4096                   68720.48         7053.15        33571.70          509.78
matmul 128x4096x4096 (batch)            68720.48         7053.15        33571.70         1911.42
elementwise relu 1M elems                 168.77            9.11          158.94           16.99
reduce sum 4096x4096->4096               1343.18           37.91          921.54           65.92
transfer 64MB                            1343.18         2689.35         5642.41          681.09
```

Sanity checks that passed: GPU beats CPU/FPGA on every compute-bound
workload (expected — FLOPS/sec ordering); the FPGA's placeholder DSP
estimate lands between CPU and GPU rather than absurdly high or low; the
64MB transfer is memory-bandwidth-bound on every device (transfer time
matches `bytesMoved / transfer_gbs`, not the compute formula).

## Results (device peak table — calibration TODO)

| Device | Peak FLOPS (TFLOPS) | Peak BW (GB/s) | Launch overhead (µs) | Source |
|--------|--------------------|-----------------|-----------------------|--------|
| CPU (c5.2xlarge) | 2.0 (placeholder) | 50 (placeholder) | 1.0 (placeholder) | TODO: measure (Phase 2 roofline) |
| GPU (A100) | 19.5 (spec sheet) | 2039 (spec sheet) | 5.0 (placeholder) | TODO: measure (Phase 3 roofline) |
| FPGA (VU9P) | 4.1 (spec estimate) | 77 (spec sheet) | 50 (placeholder) | TODO: measure (Phase 7) |
| TPU v4 | 275 (spec sheet, bf16) | 1200 (spec sheet) | 10 (placeholder) | TODO: measure (Phase 8) |

## Hardware notes
- Builds and runs anywhere (validated on Mac). Calibration requires: Linux
  CPU, GPU instance, FPGA F1, GCP TPU — each phase's roofline step
  (Phase 2 step 10, Phase 3 step 21, Phase 7 synthesis reports, Phase 8
  MXU utilization) is the source of truth this table's TODOs get filled
  from.
