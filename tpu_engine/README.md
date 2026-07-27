# Phase 8: TPU Backend

**Status: CODE COMPLETE (13/13 steps, 2026-07-27). No GCP TPU VM on this
Mac, so — same split as Phases 3/4/7 — every step is real, non-stub code:
JAX scripts written to run uncommented on a real TPU VM, an MLIR
`DialectConversion` pass (step 3) toolchain-gated one level deeper than
Phase 4 (needs StableHLO built against the exact same LLVM commit, not
just any MLIR build), and three steps (`layout_opt`, `hbm_sram`,
`cost_model`) that need no TPU/JAX at all and were actually compiled and
run locally with real captured output (`clang++`, no CMake, same
convention as `fpga_engine`'s `*_model.cpp` files). Each step's own
README documents exactly which half is measured vs. TODO.**

## Overview
JAX/XLA-based TPU backend. MLIR runtime dialect lowers to StableHLO, then to XLA
for TPU execution. Measures MXU utilization, HBM bandwidth, ICI collective bandwidth.

## Steps

| # | Directory | What | Status |
|---|-----------|------|--------|
| 1 | gcp_setup | TPU VM, JAX install, validate with matmul | code-complete, hardware-gated |
| 2 | tpu_benchmarks | MXU util, HBM bandwidth, ICI latency | code-complete, hardware-gated |
| 3 | stablehlo_lower | Runtime dialect → StableHLO lowering pass | code-complete, toolchain-gated (MLIR + StableHLO) |
| 4 | stablehlo_execute | Execute StableHLO via JAX, validate outputs | code-complete, hardware/toolchain-gated |
| 5 | layout_opt | Tile padding for systolic array alignment | **portable model run locally**, hardware half TODO |
| 6 | hbm_sram | Explicit data movement (no hardware cache) | **analytical model run locally**, hardware half TODO |
| 7 | pjit_distributed | pjit sharding across TPU chips | code-complete, hardware-gated |
| 8 | ici_collectives | TPU-native all-reduce via ICI | code-complete, hardware-gated |
| 9 | mxu_opt | 128×128 alignment for MXU saturation | code-complete, hardware-gated |
| 10 | vliw_analysis | XLA HLO instruction bundling analysis | script done + design analysis written, hardware-gated for real output |
| 11 | tpu_profiler | TPU Cloud Profiler integration | code-complete, hardware-gated |
| 12 | cost_model | $/FLOP and FLOPS/Watt: TPU vs GPU | **code-complete AND run locally** (no hardware dependency) |
| 13 | sparsecore | SparseCore embedding lookup (TPU v5 only) | code-complete, hardware/toolchain-gated (v5-only) |

## Hardware notes
- Required: GCP TPU VM (apply for TRC free quota early)
- Cheapest entry: v4-8 (8 TPU chips)
- Software: JAX, jaxlib with TPU support
- Step 3 additionally needs StableHLO built against the same LLVM commit
  as the MLIR build (`compiler/mlir_setup`) — a stricter pin than the rest
  of Phase 4, since StableHLO doesn't version against LLVM releases.
- Step 13 needs TPU v5 specifically (SparseCore doesn't exist on v4) plus
  the separate `jax-tpu-embedding` library.

## Next
Hardware validation is deferred along with every earlier phase — see the
root `CLAUDE.md`'s execution strategy. Next local-implementation phase:
Phase 9 (Inference Serving), currently stubbed.
