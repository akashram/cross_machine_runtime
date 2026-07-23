# fixed_point — ap_fixed<W,I> precision/resource/latency tradeoff

**Status: code complete — requires Vitis HLS on AWS F1 to synthesize.**

## What this measures
One templated 8x8 fixed-point matmul body (`matmul_fixed<T>` in
`fixed_point_matmul.cpp`), instantiated at three precisions as three
separate top-level kernels — same algorithm, only `ap_fixed<W,I>` differs,
so resource/latency/accuracy differences are attributable to precision
alone:

| Kernel | Type | Total bits | Integer bits | Expected profile |
|---|---|---|---|---|
| `fixed_point_matmul_8` | `ap_fixed<8,4>` | 8 | 4 | Cheapest resources, most exposed to quantization error and overflow near the representable range |
| `fixed_point_matmul_16` | `ap_fixed<16,6>` | 16 | 6 | Typical ML inference default when INT8 isn't precise enough |
| `fixed_point_matmul_32` | `ap_fixed<32,10>` | 32 | 10 | Expected to track float32 accuracy closely, still cheaper than a real FP multiplier/adder |

Accumulation happens in the same width `T` as the inputs (not a wider
intermediate), deliberately — bit growth from 8 repeated MACs per output
element is exactly where overflow/precision tradeoffs show up, and hiding
that behind a wider accumulator type would understate the narrow-precision
variant's real error.

## Results
TODO: synthesize on F1. Accuracy comparison needs a host-side float32
reference matmul over representative inputs — RMS error vs. float32 per
precision, not just resource counts.

| Kernel | DSP | LUT | FF | Latency (cycles) | Fmax (MHz) | RMS error vs. float32 |
|---|----|----|----|----|----|----|
| `_8`  | TODO | TODO | TODO | TODO | TODO | TODO |
| `_16` | TODO | TODO | TODO | TODO | TODO | TODO |
| `_32` | TODO | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: AWS F1, Vitis HLS 2022.x
- Synthesize: `vitis_hls -f run_hls.tcl` (one run per kernel)
