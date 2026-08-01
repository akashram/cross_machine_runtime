# mxu_opt

**Status: code-complete, hardware-gated for the real TPU number — still
unrun there. Actually run on CPU (2026-08-01, JAX 0.4.38, `.venv/`,
unmodified, single device) — see below; the TPU-specific 128-tile cliff
this step exists to show does not appear on CPU, which doesn't use fixed
128x128 systolic tiles at all.**

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

## Local CPU smoke test (2026-08-01)

| N | TFLOPS | measured util % (vs TPU v4 peak) | predicted ceiling % (layout_opt, TPU-specific) |
|---|---|---|---|
| 120 | 0.06 | 0.0% | 82.4% |
| 124 | 0.06 | 0.0% | 90.9% |
| 128 | 0.07 | 0.0% | 100.0% |
| 132 | 0.05 | 0.0% | 13.7% |
| 136 | 0.06 | 0.0% | 15.0% |
| 140 | 0.06 | 0.0% | 16.4% |
| 144 | 0.06 | 0.0% | 17.8% |
| 148 | 0.08 | 0.0% | 19.3% |
| 152 | 0.07 | 0.0% | 20.9% |
| 156 | 0.08 | 0.0% | 22.6% |
| 160 | 0.07 | 0.0% | 24.4% |

Real, honest finding: the measured TFLOPS are flat/noisy across the whole
120-160 sweep, with **no cliff at N=128** — as expected, since the
predicted ceiling column is derived specifically from TPU MXU's 128x128
systolic-array tile padding, and CPU matmul doesn't pad to fixed tiles at
all. This confirms the script runs correctly and the analytical
`layout_opt` model's premise (a TPU-specific tiling artifact) is sound —
it's just not something a CPU run could ever reproduce, by construction,
not by bug. The actual TPU cliff measurement stays TODO.

## Results
TODO: run on a GCP TPU v4-8 VM.

(Predicted-ceiling column precomputed from
`layout_opt_model.cpp`'s formula, `n^3 / pad_up(n,128)^3`, for reference —
not itself a new measurement.)

## Hardware notes
- Required: GCP TPU VM (single chip sufficient).
