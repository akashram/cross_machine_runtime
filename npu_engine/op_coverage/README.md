# op_coverage — NPU operator coverage analysis

**Status: code-complete AND locally run (pure classification logic, no
toolchain dependency).**

## What this measures
PLAN.md Phase 15 step 4: "a real, honest study of which of
`compiler/dialect`'s ops are NPU-deployable given the restricted op set
(conv/matmul/quantized elementwise; not arbitrary control flow), directly
informing step 6."

## Design
A hand-classified table (`op_coverage.cpp`) over all 18 ops actually
defined in `compiler/dialect/RuntimeOps.td` (read directly, not assumed),
against the restricted operator model real NPU toolchains expose (CoreML
op support tables, ONNX Runtime's NNAPI/CoreML/QNN execution providers):
fixed-function conv/matmul engines plus a limited activation/elementwise
set, no dynamic indexing, no collective communication. Each entry gets one
of five classifications (`Eligible` / `EligibleWithCaveat` / `Ineligible`
/ `Inherited` / `NotApplicable`) with a real, specific reason — see
`op_coverage.h`'s doc comment and `op_coverage.cpp`'s table for the full
per-op reasoning.

Feeds `compiler/placement/PlacementPass.cpp`'s NPU eligibility filter
(step 6): that file mirrors this table's Eligible/EligibleWithCaveat set
as a small inline check (`npuEligibleOp()`), kept in sync by comment
cross-reference rather than a shared header, since `compiler/placement`
is MLIR-gated and `npu_engine/` is deliberately dependency-free.

## Results (captured 2026-07-30, `clang++ -O2 -std=c++17 -Wall
op_coverage.cpp op_coverage_main.cpp -o op_coverage_main &&
./op_coverage_main`, this Mac)

```
18 ops total: 7 ELIGIBLE, 3 ELIGIBLE-with-caveat, 4 INELIGIBLE,
1 INHERITED (fusion_group), 3 N/A (structural/collective/already-bound)

of 14 ops given a real per-op NPU judgment (excluding N/A/Inherited),
10/14 (71%) are NPU-eligible -- gather and scatter are the only two
genuinely excluded (dynamic indexing, not activation/elementwise math,
is where NPU support actually breaks down, per SCOPE.md's restricted-
operator-model note).
```

(Full per-op table with reasons is printed by the same run — see
`op_coverage.cpp`'s `kTable` for the authoritative text; too long to
duplicate here verbatim.)

## Findings
- **Gather/scatter are the only two ops excluded on architectural
  grounds** (dynamic, data-dependent indexing) — every activation and
  elementwise op in the dialect is at least `EligibleWithCaveat`. This
  directly supports SCOPE.md's framing: NPU's restricted operator model
  is about dynamic control/indexing, not basic math.
- **A real, separate finding worth flagging**: `PlacementPass.cpp`'s
  `costKeyFor()` — read directly while writing this step — does NOT
  currently assign ANY device to `GatherOp`/`ScatterOp` for CPU/GPU/FPGA/
  TPU either; they fall through to its final `return std::nullopt` and
  are skipped by the per-op placement loop entirely, for every device,
  not just NPU. So the NPU-specific exclusion this step feeds into step 6
  isn't carving NPU out of something the other four devices already
  handle — it's a pre-existing placement-pass gap that predates Phase 15
  and applies uniformly today. Every op `PlacementPass` DOES currently
  place (matmul, conv, add, mul, sub, relu, gelu, sigmoid, softmax,
  reduce, fusion_group — 11 ops) is NPU-eligible or NPU-eligible-with-
  caveat. Documented honestly (see `op_coverage_main.cpp`'s own printed
  text) rather than overclaiming the filter is presently load-bearing.

## Platform notes
No CMake target — manually built and run per the command above.
