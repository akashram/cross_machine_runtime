#pragma once
#include <string>
#include <vector>

// PLAN.md Phase 15 step 4: operator coverage analysis. A real, honest
// classification of every op in compiler/dialect/RuntimeOps.td (read
// directly from that file — 18 ops as of this writing: matmul, conv,
// relu, gelu, sigmoid, add, mul, sub, softmax, reduce, gather, scatter,
// yield, fusion_group, transfer, all_gather, reduce_scatter,
// kernel_call) against the restricted operator model real NPU toolchains
// expose (CoreML's op support tables, ONNX Runtime's NPU execution
// providers — NNAPI/CoreML EP/QNN EP): fixed-function conv/matmul
// engines plus a limited activation/elementwise set, no dynamic
// indexing, no collective communication.
//
// Feeds step 6 (compiler/cost_model + compiler/placement integration):
// PlacementPass.cpp mirrors this table's ELIGIBLE/CAVEAT set (as a
// small inline table, since compiler/placement is MLIR-gated and can't
// portably depend on this directory) — kept in sync by comment
// cross-reference, the same pattern thermal_router.cpp/thermal_policy.cpp
// use to keep a hardware-touching file and a portable file from
// diverging in their decision logic.

namespace npu_engine {

enum class Eligibility {
  Eligible,          // real NPU toolchains support this op directly
  EligibleWithCaveat, // supported, but with a real disclosed restriction
  Ineligible,        // architecturally unsupported — falls back to CPU/GPU
  Inherited,         // eligibility depends entirely on constituent ops (fusion_group)
  NotApplicable,     // not a per-op placement decision at all (structural/collective/already-bound)
};

struct OpCoverageEntry {
  std::string op_name;
  Eligibility eligibility;
  std::string reason;
};

// The real, hand-classified table. See op_coverage.cpp's comments at each
// entry for the specific toolchain behavior being cited.
const std::vector<OpCoverageEntry> &npu_op_coverage_table();

// True only for Eligible/EligibleWithCaveat — the set PlacementPass.cpp's
// NPU candidate filter should admit an op into. Ineligible/NotApplicable/
// Inherited(unresolved) all return false: NotApplicable ops are never
// placement-pass candidates in the first place (matches costKeyFor's own
// std::nullopt cases in PlacementPass.cpp), so this function is only ever
// consulted for the ops that already reach that per-op cost-comparison loop.
bool npu_eligible(const std::string &op_name);

} // namespace npu_engine
