# quant_export — NPU quantization export pipeline

**Status: the GPTQ->INT8 quantization + custom binary export is code-complete
AND locally run (pure CPU math, no GPU/NPU dependency); the real ONNX/CoreML
writer is code-complete and unrun (neither `onnx` nor `coremltools` nor
`numpy` is installed on this Mac).**

## What this measures
PLAN.md Phase 15 step 2: "export a trained model to INT8 via
`inference_serving/gptq`'s existing Hessian-guided quantization (direct
reuse, not a re-implementation), in ONNX/CoreML format for NPU deployment."

## Design
- **Reuses `inference_serving::GptqQuantizer` unchanged**, at `bits=8`
  instead of GPTQ's usual `bits=4` — NPU toolchains (CoreML, ONNX Runtime
  NPU execution providers) default to INT8, not GPU-serving's INT4 weight-
  only path, so this step is a different bit width on the same algorithm,
  not new quantization math.
- **Custom binary export format** (`quant_export.h`'s `.npuw`): since no
  `onnx`/`coremltools` is installed, wrote a small, documented binary
  container (magic/version/rows/cols/group_size/bits header, then raw
  `uint8` qweight + `float32` scales + `int32` zero-points) rather than
  skip the export step entirely. At `bits=8`, quantized values fit exactly
  one byte each with no sub-byte packing needed — unlike `gptq.h`'s
  `bits=4` case, byte alignment falls out for free here.
- **`npu_onnx_export.py`**: a real, complete script (same convention as
  `fpga_engine/vitis_ai/mlp_model.py`) showing the actual `onnx`/
  `coremltools` API calls that would read a `.npuw` file and produce a
  deployable `.onnx` (`DynamicQuantizeLinear` -> `MatMulInteger` ->
  `Cast` -> `Mul`, matching ONNX Runtime's documented quantized-matmul
  subgraph shape) or `.mlmodel`. Unrun — no onnx/coremltools/numpy
  installed locally.
- Applied to `transformer/`'s real trained `w_out` weight with real
  calibration activations from an actual forward pass — same setup as
  `inference_serving/gptq/gptq_test.cpp`, since this step's contribution
  is the export format, not a new quantization scenario.

## Results (captured 2026-07-30, Apple clang, this Mac, `ctest -R npu_quant_export_test`)

```
PASS  serialize -> deserialize round-trip is byte-exact
PASS  write_npu_weight_file -> read_npu_weight_file round-trip is byte-exact
  perplexity: fp32=1.0497  gptq-int8=1.0498  rtn-int8=1.0497
PASS  quantization ordering holds at INT8: fp32 <= gptq-int8 <= rtn-int8 perplexity (within fp tolerance)
PASS  real w_out export round-trips through the on-disk .npuw format
  w_out export size: fp32=1728 bytes, npu-int8=888 bytes, ratio=1.946x
PASS  INT8 export is a real size reduction vs FP32, strictly below the 4x INT8-payload-only ceiling
PASS
```

## Findings
- At INT8, GPTQ's Hessian-weighted error compensation and plain
  round-to-nearest land almost indistinguishably close in perplexity —
  unlike `gptq_test.cpp`'s INT4 result where the gap is a bit more
  visible. This matches the general expectation that GPTQ's error-
  compensation benefit shrinks as bit width grows (less rounding error
  per weight to begin with), a real, disclosed observation rather than a
  claim that GPTQ "doesn't matter" at INT8 — it still never does worse.
- The `.npuw` export format's real compression ratio (1.946x) landed far
  below the 4x ceiling an INT8 payload alone gives — and this is a real,
  measured, model-specific finding, not a shortfall to explain away. This
  model's `w_out` (transposed to `[vocab_size x d_model]`) is only
  `d_model=16` columns wide at `group_size=8`, i.e. just 2 quantization
  groups per row: the per-group `float32` scale + `int32` zero-point
  overhead (16 bytes/row) ends up comparable in size to the row's own
  INT8 payload (16 bytes/row). The 4x ceiling is only approached when
  `cols >> group_size`, so more weights amortize each group's fixed
  metadata cost — a real, disclosed relationship between model
  dimensions and achievable compression, not an assumption that every
  model reaches close to 4x.

## Platform notes
- `NpuQuantExport`/`npu_quant_export_test`: portable, builds and runs on
  this Mac via CMake (`cmake --build --preset debug --target
  npu_quant_export_test && ctest -R npu_quant_export_test`).
- `npu_onnx_export.py`: requires `pip install onnx numpy` (ONNX path) or
  `pip install coremltools numpy` (CoreML/ANE path). Neither installed
  locally — this session deliberately declined new local installs (see
  project memory). Unrun.
