#include "op_coverage.h"

namespace npu_engine {

namespace {

// clang-format off
const std::vector<OpCoverageEntry> kTable = {
  {"matmul", Eligibility::Eligible,
   "Core NPU primitive -- fixed-function matrix engines exist specifically for this op "
   "(CoreML MIL 'matmul'/'linear', ONNX Runtime NNAPI/CoreML/QNN EPs all list it as a "
   "first-class supported op)."},
  {"conv", Eligibility::Eligible,
   "The NPU's other core primitive alongside matmul -- dedicated convolution engines are "
   "the defining hardware feature (SCOPE.md's 'fixed-function matrix/convolution engines' "
   "framing). Universally supported across CoreML/NNAPI/QNN."},
  {"relu", Eligibility::Eligible,
   "Trivial clamp, supported on every NPU toolchain surveyed."},
  {"gelu", Eligibility::EligibleWithCaveat,
   "Supported on CoreML/ONNX Runtime NPU EPs, but typically via a tanh/erf polynomial "
   "approximation baked into the fixed-function activation LUT rather than the exact erf "
   "formulation -- a real, small numerical deviation from the FP32 reference, same kind of "
   "disclosed approximation this repo already makes elsewhere (e.g. adversarial/'s Wald vs "
   "Clopper-Pearson bound)."},
  {"sigmoid", Eligibility::Eligible,
   "Standard LUT-backed activation, universally supported."},
  {"add", Eligibility::Eligible,
   "Quantized elementwise add -- explicitly called out in SCOPE.md's restricted-operator-model "
   "note as core NPU-supported territory."},
  {"mul", Eligibility::Eligible,
   "Quantized elementwise mul -- same as add."},
  {"sub", Eligibility::Eligible,
   "Quantized elementwise sub -- same as add."},
  {"softmax", Eligibility::EligibleWithCaveat,
   "Supported on modern NPU toolchains (needed for on-device attention/transformer inference), "
   "but was historically a common CPU-fallback op on older NPU generations/toolchain versions "
   "since it doesn't decompose into a pure matmul+activation the way the other elementwise ops "
   "do -- flagged as a caveat rather than a flat yes since toolchain-version-dependence is real, "
   "not assumed away."},
  {"reduce", Eligibility::EligibleWithCaveat,
   "Sum/mean reductions over a fixed, aligned axis set are well-supported (they're exactly what "
   "a conv/pooling engine already does); reduce with unusual/dynamic axis combinations is a "
   "common CPU-fallback case across NNAPI/CoreML -- this op's ReduceKind attribute (SUM/MAX/MEAN, "
   "see RuntimeTypes.td) means eligibility is genuinely per-instance, not per-op, hence the caveat "
   "rather than a flat Eligible."},
  {"gather", Eligibility::Ineligible,
   "Dynamic, data-dependent indexing -- the same restriction tpu_engine/stablehlo_lower's real "
   "lowering pass documents leaving as a genuine TODO (gather has no clean systolic/fixed-function "
   "mapping). NPUs' fixed dataflow engines are architecturally worse-suited to this than even TPU's "
   "MXU, since NPUs typically have no general-purpose vector unit to fall back to on-chip at all."},
  {"scatter", Eligibility::Ineligible,
   "Same reasoning as gather, and worse: write-side dynamic indexing has even less hardware support "
   "on fixed-function inference engines, which are built assuming a static, compile-time-known "
   "dataflow graph."},
  {"yield", Eligibility::NotApplicable,
   "Structural terminator op (region yield) -- never appears in PlacementPass's per-op cost "
   "comparison loop (costKeyFor returns std::nullopt for it), so device eligibility doesn't apply."},
  {"fusion_group", Eligibility::Inherited,
   "A fusion_group's NPU eligibility is exactly the AND of its constituent ops' eligibility -- "
   "if every fused op is Eligible/EligibleWithCaveat, the whole group is; if any constituent is "
   "Ineligible, the whole group must fall back (a real NPU compiler can't partially execute a "
   "fused kernel across two backends)."},
  {"transfer", Eligibility::NotApplicable,
   "Already device-bound by construction (source and destination devices are its own operands) -- "
   "the placement pass doesn't choose a device for a transfer op, it inserts one after the fact."},
  {"all_gather", Eligibility::Ineligible,
   "Collective communication has no meaning for a single edge/mobile NPU -- these ops exist for "
   "Phase 5/6's multi-node/multi-GPU distributed training, a use case NPUs (inference-only, "
   "single-device edge hardware per SCOPE.md's NPU section) are not part of."},
  {"reduce_scatter", Eligibility::Ineligible,
   "Same reasoning as all_gather."},
  {"kernel_call", Eligibility::NotApplicable,
   "Already a concrete, device-specific kernel invocation produced by kernel specialization "
   "(step 11) -- post-placement by construction, not a placement-pass decision."},
};
// clang-format on

} // namespace

const std::vector<OpCoverageEntry> &npu_op_coverage_table() { return kTable; }

bool npu_eligible(const std::string &op_name) {
  for (const auto &entry : kTable) {
    if (entry.op_name == op_name) {
      return entry.eligibility == Eligibility::Eligible ||
             entry.eligibility == Eligibility::EligibleWithCaveat;
    }
  }
  return false; // unknown op name: conservatively ineligible
}

} // namespace npu_engine
