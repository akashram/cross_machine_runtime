//===- StableHLOLowerPass.h - Phase 8 step 3: runtime -> StableHLO ------===//
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace runtime {

// Lowers every op in the `runtime` dialect (compiler/dialect) to StableHLO,
// the dialect JAX/XLA actually consume for TPU execution. Mirrors
// compiler/affine_lower's role for the CPU path: both are dialect-conversion
// passes that sit right after placement (step 9) has decided an op runs on
// this backend, translating the backend-neutral runtime dialect into
// something that backend's compiler understands.
//
// `runtime.fusion_group` is *not* lowered here — it's dissolved back into
// its constituent ops first (unlike the CPU/GPU kernel-specialization path,
// XLA does its own fusion during HLO compilation, so re-fusing before
// StableHLO would just have XLA undo or fight the earlier decision).
std::unique_ptr<mlir::Pass> createStableHLOLowerPass();

} // namespace runtime
