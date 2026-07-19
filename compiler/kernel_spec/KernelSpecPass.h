//===- KernelSpecPass.h - Step 11: lower to device-specific kernels -----===//
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace runtime {

// Terminal lowering for the `runtime` dialect: every op carrying a
// `runtime.device` attribute (set by the placement pass, step 9) is
// replaced with a `runtime.kernel_call` referencing a symbol implemented
// in cpu_engine/, gpu_engine/, or fpga_engine/ — declared in the module as
// an external `func.func private` so the reference is a real, verifiable
// symbol the AOT pipeline (step 12) resolves against actual object files
// at link time. The (op-kind, device) -> symbol lookup table lives in
// KernelSpecPass.cpp; `runtime.fusion_group` is keyed by (fusion_kind,
// device) instead, since a fused group has no single op-kind.
//
// Ops with no entry in the lookup table (i.e. this project hasn't written
// that kernel yet) are left unlowered — the pass reports how many it
// skipped rather than failing, since a partially-specialized module is
// still useful for testing the passes that ran before it.
std::unique_ptr<mlir::Pass> createKernelSpecializationPass();

} // namespace runtime
