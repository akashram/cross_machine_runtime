# HLS Dot Product Kernel

**Status: code complete — requires Vitis HLS on AWS F1 to synthesize/cosimulate.**

## What this measures
First real HLS kernel. Two variants in `dot_product.cpp`, so the II
comparison PLAN.md asks for has an actual baseline instead of a single
already-pipelined kernel:

- `dot_product_naive` — single float accumulator, no `PIPELINE` pragma.
  Expected to bottleneck on the FP adder's own pipeline latency (loop-carried
  dependency: iteration `i+1`'s add can't start until iteration `i`'s add
  result is available), not on memory or control overhead.
- `dot_product` — 4 independent partial accumulators
  (`#pragma HLS ARRAY_PARTITION ... complete` + round-robin index) break that
  dependency chain, so `#pragma HLS PIPELINE II=1` has a real shot at
  actually scheduling one MAC per cycle instead of stalling on adder
  latency. The 4 partials combine once, after the loop.

Whether 4 accumulators is enough to fully hide the adder's latency (vs.
needing 5+, matching the ~5-cycle latency of a UltraScale+ float adder)
is exactly the kind of thing this step exists to measure, not assume —
recorded in the results table below once synthesized.

## Results
TODO: synthesize + cosimulate on F1.

| Kernel | II | Fmax (MHz) | DSP | LUT | FF |
|--------|----|-----------|-----|-----|----|
| `dot_product_naive` | TODO | TODO | TODO | TODO | TODO |
| `dot_product` (4 accumulators, PIPELINE II=1) | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: AWS F1 (f1.2xlarge), Vitis HLS 2022.x
- Synthesize: `vitis_hls -f run_hls.tcl`, then `tcl_pipeline/run_pipeline.sh` for place/route
