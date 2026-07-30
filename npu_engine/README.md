# Phase 15: NPU Backend

**Status: steps 2-7 CODE COMPLETE (6/7 steps not counting step 1, which is
genuinely hardware-gated), 2026-07-30.** No AWS/GCP rental story for NPUs
(edge/mobile hardware: Apple ANE, Qualcomm Hexagon, Google Coral), so —
same split as Phases 3/7/8 — every step here is real, non-stub code:
quantization export reusing `inference_serving/gptq` unchanged,
compiler cost-model + placement-pass integration, a real ServingRouter
backend registration, and three steps (`quant_export`'s C++ half,
`cost_model`, `op_coverage`, `thermal`'s simulation half) that need no
NPU hardware or toolchain at all and were **actually compiled and run
locally** (`clang++`/CMake, real captured output in each step's own
README, same convention as `fpga_engine`/`tpu_engine`'s `*_model.cpp`
files).

## Overview
Same hardware-gated-but-locally-codeable pattern as Phase 3 (GPU), Phase 7
(FPGA), Phase 8 (TPU): NPU toolchain validation needs real ANE/Coral
hardware and `coremltools`/`onnxruntime` (neither installed on this Mac —
confirmed via `python3 -c "import coremltools"` / `import onnx`, both
`ModuleNotFoundError`), so that piece (step 1) is deferred to hardware
validation. Everything else — quantization export, cost modeling, op
eligibility, thermal modeling, and integration into the existing
compiler/serving-router code — has no such dependency and is built and
run today.

## Steps

| # | Directory | What | Status |
|---|-----------|------|--------|
| 1 | (none) | CoreML/ONNX Runtime NPU toolchain validation, trivial op on real hardware | deferred to hardware validation — no ANE/Coral hardware, no coremltools/onnxruntime installed |
| 2 | quant_export | GPTQ (reused, unchanged) INT8 export pipeline + custom `.npuw` binary format | **code-complete AND locally run** (C++ half); ONNX/CoreML writer code-complete, unrun (no onnx/coremltools) |
| 3 | cost_model | NPU vs. CPU vs. GPU INT8 latency/efficiency model | **code-complete AND locally run** (no hardware dependency) |
| 4 | op_coverage | Real per-op NPU-eligibility classification of all 18 `compiler/dialect` ops | **code-complete AND locally run** (no hardware dependency) |
| 5 | thermal | NPU/SoC power-thermal response model | **portable simulation run locally**, real SMC sensor read code-complete and unrun (no stable public per-ANE-block sensor key) |
| 6 | (compiler/cost_model, compiler/placement) | NPU device + op-eligibility filter integration | **CostModel.cpp edit code-complete AND locally run** (cost_model has no MLIR dependency); **PlacementPass.cpp edit code-complete, joins existing MLIR-toolchain-gated unrun bucket** (unchanged status, not a new gap) |
| 7 | (inference_serving/serving_backend) | NPU registered as a fifth `ServingRouter` backend, `available=false` | **code-complete AND locally run**, existing `ctest` suite still passing |

## Design highlights
- **Quantization export reuses `inference_serving::GptqQuantizer`
  unchanged**, at `bits=8` instead of GPTQ's usual `bits=4` — NPU
  toolchains default to INT8, not GPU-serving's INT4 weight-only path.
  Applied to `transformer/`'s real trained `w_out` weight with real
  calibration activations, same setup as `inference_serving/gptq`'s own
  test.
- **Operator coverage is a real, per-op classification**, not a
  hand-waved table: walks all 18 ops actually defined in
  `compiler/dialect/RuntimeOps.td`. Real finding: gather/scatter are the
  only two architecturally excluded (dynamic indexing); everything else
  is eligible or eligible-with-caveat. A SEPARATE real finding from
  reading `PlacementPass.cpp` directly: it doesn't currently place
  gather/scatter for ANY device (not just NPU) — see
  `op_coverage/README.md` and `compiler/placement/README.md` for the full
  explanation. The NPU eligibility filter added to `PlacementPass.cpp` is
  real, correct code that becomes load-bearing only once gather/scatter
  placement support exists for any device — documented honestly rather
  than overclaimed.
- **Cost model isolates efficiency from throughput**: the NPU wins on
  power efficiency (7.9 TOPS/W vs. GPU's 0.78 TOPS/W) and on tiny-workload
  absolute latency (lowest dispatch overhead of any device modeled), but
  loses to GPU on large-workload absolute latency (lower peak TOPS) — both
  real, measured outcomes from the same model, not a one-sided "NPU wins"
  narrative.
- **Thermal model reuses `fpga_engine/thermal_router`'s exact structural
  split** (pure decision logic vs. hardware-touching read), changing only
  the physically-motivated constants (smaller thermal mass -> smaller
  tau; SoC-integrated -> lower thresholds). Found and documented a real,
  disclosed platform gap: unlike FPGA's XADC, there's no stable *public*
  per-ANE-block thermal sensor on Apple Silicon.
- **ServingRouter integration places NPU last in the fallback priority
  order** (after FPGA, before CPU) — an inference-only, edge/mobile-first,
  restricted-operator-model backend is a different kind of accelerator
  than the three general datacenter ones already registered, not a peer
  competing for top fallback priority.

See `DESIGN.md` for the full rationale, including why quantization export
and thermal modeling both reuse existing code unchanged rather than
reimplementing.

## Hardware/toolchain notes
- Step 1 needs: a real Apple Silicon Mac (for CoreML/ANE) or a Coral USB
  Accelerator (for ONNX Runtime's Coral EP), plus `coremltools`/`onnx`/
  `onnxruntime`/`numpy` installed. None installed locally — this session
  deliberately declined new local installs (see project memory on "no new
  local installs").
- `npu_onnx_export.py` (step 2's ONNX/CoreML writer) needs the same
  Python packages. Unrun.
- `npu_thermal_router.cpp` (step 5's hardware half) needs IOKit access
  and a chip-generation-specific SMC key that has no stable public spec.
  Unrun.
- `PlacementPass.cpp`'s edit (step 6) joins the rest of Phase 4 in being
  unbuilt without `MLIR_DIR` set — unchanged status from before this
  phase, not a new NPU-specific gap.

## Next
Hardware validation is deferred along with every earlier phase — see the
root `CLAUDE.md`'s execution strategy. Next local-implementation item:
Phase 16's remaining local-only pieces (containers), being implemented
in parallel this same session.
