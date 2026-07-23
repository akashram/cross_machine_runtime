# dsp_lut — DSP48E2 vs LUT-fabric multiply tradeoff

**Status: code complete — requires Vitis HLS on AWS F1 to synthesize.**

## What this measures
`dsp_lut.cpp` implements the identical 256-element fixed-point
(`ap_fixed<16,4>`) multiply-accumulate twice, differing only in
`#pragma HLS BIND_OP ... impl=dsp` vs `impl=fabric` — so any resource,
timing, or power difference measured is attributable to the DSP-vs-LUT
binding choice alone, not to a different algorithm.

- `dsp_lut_dsp`: each multiply maps to a DSP48E2 block (dedicated 27x18
  multiplier + pre-adder + accumulator hardware). Expected near-zero LUT
  cost per multiply, bounded by the device's fixed DSP48E2 count (6840 on
  VU9P) — a design with enough parallel multiplies eventually runs out of
  DSPs before LUTs.
- `dsp_lut_fabric`: same multiply, forced onto general logic. Expected
  real LUT/FF cost per multiply, freeing DSP48E2 blocks for other kernels
  sharing the device and avoiding DSP cascade-routing congestion in
  designs with many small multiplies.

When to prefer each is the actual output of this step — expected to be:
DSP for a small number of wide multiplies (dot_product, ml_kernel), fabric
when DSP48E2 count is the binding constraint on a design with many
narrow/parallel multiplies and spare LUTs.

## Results
TODO: synthesize on F1.

| Variant | DSP | LUT | FF | Fmax (MHz) | Power (W) |
|---|----|----|----|----|----|
| `dsp_lut_dsp` | TODO | TODO | TODO | TODO | TODO |
| `dsp_lut_fabric` | TODO | TODO | TODO | TODO | TODO |

## Hardware notes
- Required: AWS F1, Vitis HLS 2022.x
- Synthesize: `vitis_hls -f run_hls.tcl`
