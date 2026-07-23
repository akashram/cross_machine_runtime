# loop_opt — UNROLL / PIPELINE / DATAFLOW comparison

**Status: code complete — requires Vitis HLS on AWS F1 to synthesize.**

## What this measures
The same 256-element block-wise multiply-accumulate (`out[i] = a[i]*b[i] +
c[i]`) implemented four ways in `loop_opt.cpp`, isolating each pragma's
effect instead of measuring them stacked together:

| Variant | Mechanism | Expected tradeoff |
|---|---|---|
| `loop_opt_baseline` | no pragma | Vitis HLS default scheduling — one loop trip fully retires before the next starts. Baseline for everything else. |
| `loop_opt_pipeline` | `PIPELINE II=1` | Overlaps loop *iterations* in the same hardware — new work can start every cycle even though a single iteration's latency is unchanged. Costs control logic (in-flight iteration tracking), not extra arithmetic units. |
| `loop_opt_unroll` | `UNROLL factor=4` | Duplicates the multiply/add datapath 4x, runs 4 iterations of logic per loop trip. Costs 4x LUT/DSP for the arithmetic; without PIPELINE, does *not* by itself give cycle-level overlap. |
| `loop_opt_dataflow` | `DATAFLOW` across 3 stages (load/compute/store) connected by `hls::stream` | Pipelines across **function calls**: stage N+1 can start consuming stage N's output before stage N finishes the whole block. Costs FIFO depth (BRAM) between stages instead of duplicated arithmetic. |

The open question this step exists to answer: for this specific workload,
does DATAFLOW's stage-level overlap beat PIPELINE's iteration-level
overlap, or does the FIFO handshake overhead between only 3 stages (vs. a
long, deep pipeline) eat the win? That's a measured answer, not assumed —
recorded below once synthesized.

## Results
TODO: synthesize on F1.

| Variant | II | Latency (cycles) | Fmax (MHz) | DSP | LUT | FF | BRAM |
|---|----|----|----|----|----|----|----|
| baseline | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| pipeline | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| unroll x4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| dataflow | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: AWS F1, Vitis HLS 2022.x
- Synthesize: `vitis_hls -f run_hls.tcl` (one run per top function)
