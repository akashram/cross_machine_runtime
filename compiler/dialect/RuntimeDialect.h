//===- RuntimeDialect.h - `runtime` dialect + attribute helpers ---------===//
//
// Steps 2-3 of Phase 4. See RuntimeOps.td for the op set and
// compiler/DESIGN.md for the design rationale (why device/shard are
// discardable attributes rather than part of the tensor type).
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Generated enum/attribute declarations from RuntimeTypes.td.
#include "RuntimeEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "RuntimeAttrDefs.h.inc"

// Generated dialect class declaration from RuntimeDialect.td.
#include "RuntimeDialect.h.inc"

// Generated op class declarations from RuntimeOps.td.
#define GET_OP_CLASSES
#include "RuntimeOps.h.inc"

namespace runtime {

// Well-known discardable attribute names set by later passes. Declared once
// here so every pass (placement, sharding, kernel_spec, cost_model callers)
// references the same string instead of re-typing "runtime.device".
inline constexpr llvm::StringLiteral kDeviceAttrName = "runtime.device";
inline constexpr llvm::StringLiteral kShardAttrName = "runtime.shard";

// Returns the op's `runtime.device` attribute, or DeviceKind::Unassigned if
// the placement pass (step 9) hasn't run yet / skipped this op.
DeviceKind getAssignedDevice(mlir::Operation *op);

// Sets `runtime.device` on `op`. Used by the placement pass and by the AOT
// pipeline's tests to construct pre-placed IR.
void setAssignedDevice(mlir::Operation *op, DeviceKind device);

// True for ops the fusion pass (step 5) is allowed to pull into a
// runtime.fusion_group: pure, single-result-or-tuple, no side effects.
bool isFusable(mlir::Operation *op);

} // namespace runtime
