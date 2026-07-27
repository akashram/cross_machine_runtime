# mxu_opt

**Status: code-complete, hardware-gated — real JAX script, unrun. No TPU
device on this Mac.**

## What this measures

PLAN.md Phase 8 step 9: matmul dimensions aligned to 128x128, utilization
%, the performance cliff at non-aligned sizes.

## Design

- `mxu_bench.py` sweeps N densely across a single 128-multiple boundary
  (120..160, step 4) rather than the coarse power-of-two sweep
  `tpu_benchmarks/mxu_util_bench.py` (step 2) already does — the point
  here is the *shape* of the cliff right at the boundary, not the broad
  behavior across scales.
- Each matmul is batched 32-fold (`einsum("bij,bjk->bik")`) so a call at
  small N still does enough work to amortize host dispatch overhead;
  without that, small-N timing would be dominated by launch latency, not
  MXU behavior, and the cliff would be invisible under that noise.
- Prints both the measured utilization % and
  `layout_opt/layout_opt_model.cpp`'s analytical padding-ceiling %
  side by side, so the real measurement (once run) is checked directly
  against the earlier model's prediction rather than left as two
  disconnected numbers in two READMEs.

## Results
TODO: run on a GCP TPU v4-8 VM.

| N | measured util % | predicted ceiling % (layout_opt) |
|---|---|---|
| 120 | TODO | 82.4% |
| 124 | TODO | 90.9% |
| 128 | TODO | 100.0% |
| 132 | TODO | 13.7% |
| 136 | TODO | 15.0% |
| ... | TODO | ... |

(Predicted-ceiling column precomputed from
`layout_opt_model.cpp`'s formula, `n^3 / pad_up(n,128)^3`, for reference —
not itself a new measurement.)

## Hardware notes
- Required: GCP TPU VM (single chip sufficient).
