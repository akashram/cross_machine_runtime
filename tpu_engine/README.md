# Phase 8: TPU Backend

**Status: STUB — requires GCP TPU VM (v4-8 slice or TRC free quota).**

## Overview
JAX/XLA-based TPU backend. MLIR runtime dialect lowers to StableHLO, then to XLA
for TPU execution. Measures MXU utilization, HBM bandwidth, ICI collective bandwidth.

## Steps

| # | Directory | What |
|---|-----------|------|
| 1 | gcp_setup | TPU VM, JAX install, validate with matmul |
| 2 | tpu_benchmarks | MXU util, HBM bandwidth, ICI latency |
| 3 | stablehlo_lower | Runtime dialect → StableHLO lowering pass |
| 4 | stablehlo_execute | Execute StableHLO via JAX, validate outputs |
| 5 | layout_opt | Tile padding for systolic array alignment |
| 6 | hbm_sram | Explicit data movement (no hardware cache) |
| 7 | pjit_distributed | pjit sharding across TPU chips |
| 8 | ici_collectives | TPU-native all-reduce via ICI |
| 9 | mxu_opt | 128×128 alignment for MXU saturation |
| 10 | vliw_analysis | XLA HLO instruction bundling analysis |
| 11 | tpu_profiler | TPU Cloud Profiler integration |
| 12 | cost_model | $/FLOP and FLOPS/Watt: TPU vs GPU |
| 13 | sparsecore | SparseCore embedding lookup (TPU v5 only) |

## Hardware notes
- Required: GCP TPU VM (apply for TRC free quota early)
- Cheapest entry: v4-8 (8 TPU chips)
- Software: JAX, jaxlib with TPU support
