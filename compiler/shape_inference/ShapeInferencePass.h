//===- ShapeInferencePass.h - Step 4: propagate tensor shapes -----------===//
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace runtime {

// Propagates ranked/static shapes through a `runtime` dialect function body.
// Runs as a fixed-point worklist: an op is revisited whenever an operand's
// shape changes, until no op's result type changes in a full pass (handles
// symbolic/dynamic dims that only resolve once an upstream producer's shape
// is itself resolved — e.g. a matmul feeding a matmul).
std::unique_ptr<mlir::Pass> createShapeInferencePass();

} // namespace runtime
