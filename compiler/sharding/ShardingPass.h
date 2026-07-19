//===- ShardingPass.h - Step 10: GSPMD-style auto-sharding ---------------===//
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace runtime {

// Canned sharding recipes, applied at `runtime.matmul` ops — the Megatron-
// style building blocks Phase 6 (distributed GPU training) composes into
// column/row-parallel linear layers. A full GSPMD-style *search* over
// sharding assignments (propagate-and-choose-the-cheapest, as in the GSPMD
// paper) is out of scope here: this pass applies one strategy uniformly,
// which is enough to validate the mechanics (attribute propagation,
// collective insertion) before Phase 6 needs the real thing.
enum class ShardingStrategy {
  Replicated,           // no sharding; sanity-check baseline
  DataParallel,         // shard the leading (batch/token) dim of lhs
  TensorParallelColumn, // shard rhs's output-feature dim (Megatron col-parallel)
  TensorParallelRow,    // shard the contraction dim of both operands
};

// Annotates every `runtime.matmul` with a `runtime.shard` (ShardSpecAttr)
// on its operands and result per `strategy`, and inserts
// `runtime.all_gather` / `runtime.reduce_scatter` wherever a strategy
// requires reconciling a sharded intermediate (e.g. row-parallel's partial
// sums across the contraction dim).
std::unique_ptr<mlir::Pass> createShardingPass(int num_devices, ShardingStrategy strategy);

} // namespace runtime
