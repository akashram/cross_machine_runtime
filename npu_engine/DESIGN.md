# NPU Backend — Design

## 1. Why NPU gets a "portable model + hardware-gated kernel" split, and why that split runs deeper here than for GPU/FPGA/TPU

Phases 3/7/8 all hit the same wall: real accelerator hardware isn't on
this Mac, so each phase writes real code for every step and defers
running the hardware-touching half. NPU hits a second, sharper version of
that wall: unlike GPU/FPGA/TPU, there is no straightforward AWS/GCP rental
story for NPU hardware at all (see SCOPE.md's NPU section) — it's
edge/mobile silicon (Apple ANE, Qualcomm Hexagon, Google Coral), not
something a cloud instance type exposes. So where Phase 3/7/8's "hardware
validation pass" table (CLAUDE.md) has a concrete AWS/GCP instance type to
provision, Phase 15's hardware validation more realistically means buying
a ~$60 Coral USB Accelerator or using this session's own Apple Silicon Mac
directly — a genuinely different kind of "next step," documented honestly
rather than forced into the existing cloud-rental table.

A second, independent gap compounds this: even the Python toolchain layer
(`coremltools`, `onnx`, `onnxruntime`, `numpy`) isn't installed locally,
confirmed directly (`python3 -c "import coremltools"` /
`import onnx` / `import numpy`, all `ModuleNotFoundError`) rather than
assumed. This session declined new local installs (see project memory),
so every piece of this phase that would need those packages is written as
real, complete, unrun code — same convention as
`fpga_engine/vitis_ai/mlp_model.py` — rather than skipped or faked.

## 2. Quantization export reuses GPTQ unchanged, at a different bit width, not a new implementation

`inference_serving/gptq`'s `GptqQuantizer` already does the real,
non-trivial work (Hessian-guided greedy column quantization with
error compensation). NPU deployment's default precision is INT8, not
GPU-serving's INT4 weight-only path — but that's a `bits` constructor
argument, not new math. `quant_export/quant_export.cpp` calls the
existing class at `bits=8` and adds exactly one new thing: a
serialization format, since no ONNX/CoreML writer library exists locally.
At `bits=8`, the quantized values happen to be byte-aligned already (no
sub-byte packing needed) — a genuine simplification GPTQ's usual `bits=4`
case doesn't get, not a shortcut taken to avoid real work.

The `.npuw` binary format is deliberately minimal (magic/version/
dimensions header, then raw bytes) — it exists so `quant_export_test.cpp`
can prove a full serialize -> deserialize -> real-model round trip is
byte-exact, which is what actually validates the export pipeline's
correctness. `npu_onnx_export.py` is the real ONNX/CoreML writer this
format would feed into once those libraries exist; see its own
docstring for why the (rows, num_groups) scale/zero-point grid gets
folded to one scale per output row for the ONNX path specifically
(`MatMulInteger` + `DequantizeLinear` only cleanly support one scale
axis without a custom op) — a disclosed simplification of the ONNX
export, not of the quantization itself, which keeps its full per-group
granularity in the `.npuw` file.

## 3. Cost model isolates the honest NPU case: efficiency, not universal speed

It would be easy to write an NPU cost model that just shows the NPU
"winning" — pick a tiny workload, or an unfairly low-power GPU comparator,
and the NPU comes out ahead on every axis. `npu_cost_model.cpp`
deliberately runs two workloads at very different scales specifically to
surface the real crossover: the NPU wins on tiny-workload absolute latency
(lowest dispatch overhead of any device modeled — no OS-mediated kernel
launch queue) AND on power efficiency (TOPS/Watt) at every scale, but
loses to the GPU on large-workload absolute latency, because its peak
TOPS (15.8) is genuinely far below the GPU's (312). This is the honest
finding SCOPE.md's NPU section is actually describing — "trade peak FLOPS
... for FLOPS/Watt at the edge" — measured directly rather than assumed,
in the same spirit as `fpga_engine/vitis_ai/dpu_vs_custom_model.cpp`'s
"a shared engine pays fixed overhead a point-design skips" finding, but
for the opposite conclusion (NPU beats a bigger accelerator on
efficiency and small-batch latency, not on raw throughput).

## 4. Operator coverage feeds placement, and honestly reports where it currently has no effect

`op_coverage/op_coverage.cpp` classifies all 18 real ops in
`compiler/dialect/RuntimeOps.td`. The interesting finding isn't just
"gather/scatter are excluded" (expected, and the whole point of the
step) — it's what reading `PlacementPass.cpp`'s `costKeyFor()` directly
revealed while building step 6: that function already returns
`std::nullopt` for `GatherOp`/`ScatterOp` today, for every device, not
just NPU. So the NPU-specific exclusion filter added in
`PlacementPass.cpp` (`npuEligibleOp()`) is real, correct code — but has
zero currently-placed ops to exclude, since the two ops it would exclude
were never placed by any device to begin with. Rather than paper over
this ("NPU excludes gather/scatter!") or quietly skip writing the filter
("it wouldn't do anything yet"), both `op_coverage/README.md` and
`compiler/placement/README.md` state the finding plainly: the filter is
real and becomes load-bearing the moment gather/scatter placement support
is added for any device, but is a no-op against `PlacementPass`'s current
op coverage. This is the same "measured, not assumed, and disclosed
either way" standard the rest of this repo holds (e.g. `rag/DESIGN.md`'s
recall/quality tradeoff, `adversarial/DESIGN.md`'s robustness/accuracy
cost).

## 5. Thermal model mirrors FPGA's exact structure, changes only the physics

`npu_thermal_policy.h`/`.cpp` and `npu_thermal_sim.cpp` are line-for-line
structural copies of `fpga_engine/thermal_router`'s two-file split (pure
decision function vs. hardware-touching read, portable RC-step-response
simulation) — deliberately, since the DECISION logic (three-tier
temperature -> allocation-fraction policy) doesn't depend on which
accelerator it's protecting. What changes are the constants that reflect
a real physical difference: an integrated mobile/edge SoC has far less
thermal mass than a discrete FPGA card with its own heatsink, so its
modeled thermal time constant is smaller (tau=4s vs. 15s) and its
response times land in single-digit seconds rather than tens of seconds
for an equivalent steady-state overshoot — a real, structural consequence
the simulation surfaces without any new logic, purely from swapping in
different physically-motivated numbers.

Writing `npu_thermal_router.cpp` (the hardware-touching half) surfaced a
real, disclosed platform gap worth stating plainly: FPGA's XADC exposes a
documented, stable per-die-region sensor API; Apple Silicon's SMC exposes
only a community-reverse-engineered, chip-generation-varying key for
overall SoC temperature, with no equivalently stable *public* per-ANE-block
reading. This is a genuine limitation in NPU thermal observability
compared to FPGA, not a missing-toolchain stand-in.

## 6. ServingRouter: NPU is a real backend, ranked honestly

`inference_serving/serving_backend/serving_router.h/.cpp` already has a
working, tested fallback-priority pattern for GPU/TPU/FPGA/CPU. Adding
NPU means answering a real design question: where does it rank? NPU
toolchains are inference-only, edge/mobile-first, and restricted-
operator-model hardware (this phase's own `op_coverage` finding) — a
categorically different kind of accelerator than the three general
datacenter backends already registered, not a peer competing with them
for priority. It's placed last among the four accelerators (after FPGA,
before CPU) in `kPriorityOrder`, and `serving_router_test.cpp` gained a
real test (`test_npu_outranks_cpu_but_not_fpga_in_fallback`) proving both
halves of that ranking claim, not just asserting it in a comment.

## 7. What's genuinely deferred, and why

- **Step 1 (toolchain validation)**: needs real ANE/Coral hardware plus
  `coremltools`/`onnxruntime`, neither available. No portable substitute
  exists for "run a trivial op on real hardware" — unlike the other
  steps, there's no arithmetic model that stands in for this one.
- **`npu_onnx_export.py`**: needs `onnx`/`coremltools`/`numpy`. Written
  complete and unrun (see item 2 above).
- **`npu_thermal_router.cpp`**: needs IOKit access and an undocumented,
  chip-generation-specific SMC key (see item 5 above) — a deeper gap than
  "no toolchain installed," since even with IOKit access there is no
  single correct key to hard-code without testing on specific target
  hardware.
- **`PlacementPass.cpp`'s edit**: joins the rest of Phase 4 in being
  unbuilt without `MLIR_DIR` — this was already true before Phase 15;
  adding NPU to it doesn't change its buildable status, and isn't treated
  as a new gap.
