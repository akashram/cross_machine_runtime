# HLS Dot Product Kernel

**Status: STUB — requires Vitis HLS on AWS F1.**

## What this measures
First HLS kernel: measure II (initiation interval) before and after
`#pragma HLS PIPELINE`, target II=1. Document resource utilization
(DSP, LUT, FF, BRAM) and Fmax.

## Results
TODO: synthesize on F1.

| Pragma | II | Fmax (MHz) | DSP | LUT | FF |
|--------|----|-----------|-----|-----|----|
| No pragma | TODO | TODO | TODO | TODO | TODO |
| PIPELINE II=1 | TODO | TODO | TODO | TODO | TODO |
| PIPELINE + UNROLL=4 | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: AWS F1 (f1.2xlarge), Vitis HLS 2022.x
- Synthesize: `vitis_hls -f synth.tcl`
