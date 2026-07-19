//===- PlacementPass.h - Step 9: cost-model-driven device placement -----===//
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace runtime {

// Assigns every placeable op (matmul, conv, elementwise, reduce, and
// fusion_group) a `runtime.device` attribute by greedily minimizing
// per-op cost: for each candidate device, cost = compute time on that
// device (CostModel::estimate_us) + sum of transfer-in costs for any
// operand whose producer landed on a different device. Inserts an
// explicit `runtime.transfer` op wherever a chosen device differs from an
// operand's source device, so the IR downstream (kernel specialization,
// step 11) never has to re-derive placement — it's fully explicit in the
// IR after this pass.
//
// This is greedy, not globally optimal: op N's placement doesn't consider
// how it constrains op N+1's transfer cost. A real cost-based scheduler
// (dynamic programming over the op DAG, or XLA-style ILP) is future work
// once there's a real workload to validate the greedy choice against.
std::unique_ptr<mlir::Pass> createPlacementPass();

} // namespace runtime
