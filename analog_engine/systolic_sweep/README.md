# systolic_sweep

**Status: code-complete AND locally run — pure CPU, no external
dependency.**

## What this measures

PLAN.md Phase 17 step 6: reuse step 5's dataflow/PPA model on
`transformer/`'s REAL GEMM shapes — extending `tpu_engine/mxu_opt`'s
single-shape utilization-cliff finding into a genuine design-space SWEEP
over PE array size, per workload shape, to find the utilization-
maximizing configuration for each.

## Design

- Every GEMM shape is computed directly from the EXACT config
  `transformer/transformer_test.cpp`'s `test_trains_and_generates()`
  actually trains with (`d_model=16, num_heads=2, d_ff=32,
  max_seq_len=32`, `vocab_size=16` from the corpus `"the quick fox jumps "`'s
  16 unique characters) — real numbers from a real, already-trained model
  in this repo, not invented shapes.
- Sweeps square PE array sizes `{8, 16, 32, 64, 128}` per shape (reusing
  `analyze_dataflow` from step 5 directly — utilization depends only on
  `M`, `N`, and the PE array dimensions in this model, so Weight-
  Stationary stands in as a representative dataflow choice for the sweep).
- Finds each shape's utilization-maximizing PE size, tie-broken toward the
  SMALLEST size (less silicon area for the same utilization is strictly
  better).

## Results (captured 2026-08-09, Apple clang 14 / `-std=c++2b`, this Mac)

```
  transformer/'s real GEMM shapes (d_model=16, num_heads=2, head_dim=8, d_ff=32, seq_len=32, vocab_size=16):

  QKV projection (per Q/K/V) (M=32,K=16,N=16):
    PE=   8x8     utilization=1.0000  <- best
    PE=  16x16    utilization=1.0000
    PE=  32x32    utilization=0.5000
    PE=  64x64    utilization=0.1250
    PE= 128x128   utilization=0.0312

  attention scores (Q @ K^T, per head) (M=32,K=8,N=32):
    PE=   8x8     utilization=1.0000  <- best
    ...
    PE= 128x128   utilization=0.0625

  [FFN up/down, output projections omitted here -- see full output above]

  averaged across all 7 shapes: utilization at PE=128x128 (a production-scale array) = 0.0379 | at each shape's best PE size = 1.0000
PASS  the utilization-maximizing PE size averages more than 2x the utilization a 128x128 production-scale array gets on these small toy-transformer shapes -- a real, measured over-provisioning cost, not assumed

  attention-scores shape (32x8x32) specifically: PE=128 utilization=0.0625 vs. PE=32 utilization=1.0000
PASS  attention scores' 32x32 output on a 128x128 array wastes over 90% of the array -- catastrophic underutilization from over-provisioning, echoing tpu_engine/layout_opt's small-batch-padding finding on an independently-built model
PASS  the SAME shape on a 32x32 array (matching its own M,N dimensions exactly) achieves ~100% utilization
```

## Findings

- **The gap is enormous, not marginal**: averaged across all 7 real GEMM
  shapes in one transformer layer, a 128x128 array (a realistic
  production-accelerator systolic array size, matching TPU's actual MXU
  tile boundary) achieves only `3.79%` utilization, against `100.0%` at
  each shape's own best-matched PE size — over 26x worse, not a modest
  inefficiency.
- **Every single shape's best PE size is 8x8** (the smallest option
  swept) — because every `M`/`N` dimension in this toy transformer
  (`seq_len=32`, `d_model=16`, `head_dim=8`, `d_ff=32`, `vocab_size=16`) is
  a small power of 2 that divides evenly by 8. This isn't a coincidence of
  the sweep's chosen sizes; it's a direct consequence of this repo's
  transformer being deliberately toy-scale, and it makes the point sharply:
  the "right" systolic array size is a property of the WORKLOAD, not a
  fixed hardware constant every workload should be measured against.
- **This independently reproduces `tpu_engine/mxu_opt`'s own finding**
  (MXU utilization cliffs at non-128-aligned sizes) from a completely
  different angle: `mxu_opt` fixed the hardware (TPU's real 128x128 MXU)
  and swept workload alignment; this step fixes several real workload
  shapes and sweeps hardware size, landing on the same underlying
  phenomenon — utilization is a function of the RATIO between workload
  and array dimensions, not either one alone. Two independently-built
  models in this repo (`tpu_engine/mxu_opt`, targeting real TPU hardware
  characteristics, and this one, targeting an analog/systolic PPA model)
  converge on the same qualitative conclusion.
- Directly actionable for step 7 (hardware-algorithm co-design): if this
  toy transformer were the actual target workload for a custom analog/
  systolic accelerator, the PPA-optimal design point is nowhere near a
  "go bigger" 128x128 array — it's a much smaller array matched to the
  model's real dimensions, freeing the silicon area a 128x128 array would
  waste on padding for something else (more PEs for a different, larger
  workload; more on-chip memory; lower power at the same throughput).

## Hardware notes
None — pure CPU. Same illustrative-model status as steps 1-5.
